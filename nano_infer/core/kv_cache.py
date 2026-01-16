import numpy as np
from .tensor import Tensor
from ..ops import functional as F
from ..backend import _nano_infer

Device = _nano_infer.Device

class StaticKVCache:
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, device=Device.CUDA):
        self.shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        self.k_cache = Tensor(np.zeros(self.shape, dtype=np.float32), device=device)
        self.v_cache = Tensor(np.zeros(self.shape, dtype=np.float32), device=device)
        self.current_seq_len = 0

    def update(self, k_new: Tensor, v_new: Tensor, start_pos: int):
        # Trigger the optimized kernel to perform transposed copy into cache.
        F.kv_cache_update(self.k_cache, self.v_cache, k_new, v_new, start_pos)
        self.current_seq_len = max(self.current_seq_len, start_pos + k_new.shape[1])

    def get_view(self):
        return self.k_cache, self.v_cache, self.current_seq_len

    def reset(self):
        self.current_seq_len = 0

class BlockAllocator:
    def __init__(self, num_blocks):
        self.free_blocks = list(range(num_blocks))
        
    def allocate(self, num):
        if len(self.free_blocks) < num:
            raise RuntimeError("Out of memory")
        ret = self.free_blocks[:num]
        self.free_blocks = self.free_blocks[num:]
        return ret
        
    def free(self, blocks):
        self.free_blocks.extend(blocks)

class PagedCacheEngine:
    def __init__(self, config, block_size=16, max_num_blocks=1024, device=Device.CUDA):
        self.block_size = block_size
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.max_batch_size = config.max_batch_size
        self.device = device
        
        self.allocator = BlockAllocator(max_num_blocks)
        
        # Physical Cache: [NumBlocks, H, BlockSize, D]
        self.k_caches = [Tensor.empty((max_num_blocks, self.n_heads, block_size, self.head_dim), device=device) for _ in range(self.n_layers)]
        self.v_caches = [Tensor.empty((max_num_blocks, self.n_heads, block_size, self.head_dim), device=device) for _ in range(self.n_layers)]
        
        # Metadata
        # block_tables: batch_idx -> list of block indices
        self.block_tables_dict = {i: [] for i in range(self.max_batch_size)}
        self.context_lens_cpu = np.zeros((self.max_batch_size,), dtype=np.int32)
        
        self.current_pos = 0 # To satisfy API, though we use context_lens_cpu
        
        # Tensors for kernels (updated lazily)
        self.block_tables_tensor = None
        self.context_lens_tensor = None
        
        # Slot mapping for current step
        self.slot_mapping = None

    def _update_metadata_tensors(self):
        # Convert block tables to tensor
        max_blocks = max(len(b) for b in self.block_tables_dict.values()) if self.block_tables_dict else 0
        max_blocks = max(max_blocks, 1)
        
        bt_np = np.zeros((self.max_batch_size, max_blocks), dtype=np.int32)
        for i, blocks in self.block_tables_dict.items():
            if blocks:
                bt_np[i, :len(blocks)] = blocks
                
        self.block_tables_tensor = Tensor(bt_np, device=self.device)
        self.context_lens_tensor = Tensor(self.context_lens_cpu, device=self.device)

    def update(self, layer_idx, k_new, v_new):
        # k_new: [B, Seq, H, D]
        B, Seq, H, D = k_new.shape
        
        # 1. Allocation (Only for first layer to avoid redundancy)
        if layer_idx == 0:
            for b in range(B):
                needed_len = self.context_lens_cpu[b] + Seq
                current_blocks = len(self.block_tables_dict[b])
                needed_blocks = (needed_len + self.block_size - 1) // self.block_size
                
                if needed_blocks > current_blocks:
                    new_blocks = self.allocator.allocate(needed_blocks - current_blocks)
                    self.block_tables_dict[b].extend(new_blocks)
            
            # Prepare slot mapping for reshape_and_cache
            # slot_mapping: [B * Seq] -> physical index
            slots = []
            for b in range(B):
                start_len = self.context_lens_cpu[b]
                for i in range(Seq):
                    curr = start_len + i
                    block_idx = self.block_tables_dict[b][curr // self.block_size]
                    block_offset = curr % self.block_size
                    slots.append(block_idx * self.block_size + block_offset)
            
            self.slot_mapping = Tensor(np.array(slots, dtype=np.int32), device=self.device)
            
            # Update context lengths immediately so kernel sees current tokens
            self.context_lens_cpu[:B] += Seq
            
            self._update_metadata_tensors()
            
        # 2. Cache Update
        # Flatten k_new, v_new to [B*Seq, H, D]
        # Note: view() is in-place, so we must restore shape afterwards
        original_k_shape = k_new.shape
        original_v_shape = v_new.shape
        
        k_new.view(B * Seq, H, D)
        v_new.view(B * Seq, H, D)
        
        F.reshape_and_cache(
            k_new, v_new,
            self.k_caches[layer_idx], self.v_caches[layer_idx],
            self.slot_mapping
        )
        
        # Restore shapes
        k_new.view(*original_k_shape)
        v_new.view(*original_v_shape)

    def get_view(self, layer_idx):
        # Return internal structures for paged attention
        return (
            self.k_caches[layer_idx],
            self.v_caches[layer_idx],
            self.block_tables_tensor,
            self.context_lens_tensor
        )

    def advance(self, num_tokens):
        # Assuming all batch items advance by same amount (or logic handles it)
        # Context lens are now updated in update()
        self.current_pos += num_tokens

    def reset(self):
        self.current_pos = 0
        self.context_lens_cpu.fill(0)
        # Free blocks
        for b in self.block_tables_dict:
            if self.block_tables_dict[b]:
                self.allocator.free(self.block_tables_dict[b])
                self.block_tables_dict[b] = []

class CacheEngine:
    def __init__(self, config, device=Device.CUDA):
        self.n_layers = config.n_layers
        self.current_pos = 0
        head_dim = config.hidden_size // config.n_heads
        # Maintain a list of caches, one per transformer layer.
        self.layers = [
            StaticKVCache(config.max_batch_size, config.max_position_embeddings, 
                          config.n_heads, head_dim, device) 
            for _ in range(config.n_layers)
        ]

    def update(self, layer_idx, k_new, v_new):
        self.layers[layer_idx].update(k_new, v_new, self.current_pos)

    def get_view(self, layer_idx):
        return self.layers[layer_idx].get_view()

    def advance(self, num_tokens):
        self.current_pos += num_tokens

    def reset(self):
        self.current_pos = 0
        for layer in self.layers:
            layer.reset()