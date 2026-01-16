import math
import numpy as np
import nano_infer.nn as nn
from nano_infer.ops import functional as F
from ..core.tensor import Tensor
from ..core.kv_cache import CacheEngine

class LlamaConfig:
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=11008,
        n_layers=32,
        n_heads=32,
        vocab_size=32000,
        rms_norm_eps=1e-6,
        max_position_embeddings=2048,
        max_batch_size=1
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.max_batch_size = max_batch_size

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_position_embeddings).astype(np.float32)
        freqs = np.outer(t, inv_freq)
        self.cos_cached = nn.Parameter(Tensor(np.cos(freqs)), requires_grad=False)
        self.sin_cached = nn.Parameter(Tensor(np.sin(freqs)), requires_grad=False)

    def forward(self, seq_len=None):
        # Return full precomputed cos and sin tables.
        # The slicing/indexing is handled by the RoPE kernel using start_pos.
        return self.cos_cached, self.sin_cached

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU activation: down_proj(SiLU(gate_proj(x)) * up_proj(x)).
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = args.n_heads
        self.head_dim = args.hidden_size // args.n_heads
        
        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.o_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, args.max_position_embeddings)

    def forward(self, x: Tensor, cache_engine: CacheEngine = None):
        B, Seq, _ = x.shape
        xq = self.q_proj(x).view(B, Seq, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(B, Seq, self.n_heads, self.head_dim)
        xv = self.v_proj(x).view(B, Seq, self.n_heads, self.head_dim)
        
        cos, sin = self.rotary_emb(Seq)
        start_pos = cache_engine.current_pos if cache_engine else 0
        F.rope(xq, xk, cos, sin, start_pos)
        
        if cache_engine is not None:
            cache_engine.update(self.layer_idx, xk, xv)
            
            # Support for PagedAttention
            if hasattr(cache_engine, 'block_tables_tensor'):
                if Seq > 1:
                    # Prefill: Standard attention on the input (assuming causal self-attention)
                    keys, values = xk.transpose(1, 2), xv.transpose(1, 2)
                else:
                    # Decoding: Paged Attention
                    k_cache, v_cache, block_tables, context_lens = cache_engine.get_view(self.layer_idx)
                    out = Tensor.empty(xq.shape, xq.dtype, device=xq.device)
                    # Flatten B and Seq=1 for kernel
                    xq_sq = xq.view(B, self.n_heads, self.head_dim)
                    sm_scale = 1.0 / math.sqrt(self.head_dim)
                    
                    F.paged_attention(out, xq_sq, k_cache, v_cache, block_tables, context_lens, sm_scale)
                    
                    # Output projection
                    output = out.view(B, Seq, -1)
                    return self.o_proj(output)
            else:
                # Static Cache
                keys, values, cur_len = cache_engine.get_view(self.layer_idx)
                # Fetch valid prefix from the static cache.
                # keys, values = keys[:, :, :cur_len, :], values[:, :, :cur_len, :]
                keys = keys.slice(2, 0, cur_len)
                values = values.slice(2, 0, cur_len)
        else:
            keys, values = xk.transpose(1, 2), xv.transpose(1, 2)
        
        xq = xq.transpose(1, 2)
        keys_T = keys.transpose(-1, -2)
        scores = F.matmul(xq, keys_T) / math.sqrt(self.head_dim)
        if scores.shape[-2] > 1:
            F.apply_causal_mask(scores, start_pos)
        probs = F.softmax(scores, dim=-1)
        output = F.matmul(probs, values).transpose(1, 2).contiguous().view(B, Seq, -1)
        return self.o_proj(output)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.attention = LlamaAttention(args, layer_idx)
        self.mlp = LlamaMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(self, x: Tensor, cache_engine: CacheEngine = None):
        h = x + self.attention(self.input_layernorm(x), cache_engine)
        return h + self.mlp(self.post_attention_layernorm(h))

class LlamaModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(args, i) for i in range(args.n_layers)])
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(self, input_ids: Tensor, cache_engine: CacheEngine = None):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, cache_engine)
        if cache_engine:
            cache_engine.advance(input_ids.shape[1])
        return self.norm(h)