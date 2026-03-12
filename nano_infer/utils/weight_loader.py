import os
import json
import numpy as np
from typing import Dict, Any, List
try:
    from safetensors.numpy import load_file
except ImportError:
    load_file = None
try:
    from safetensors.torch import load_file as load_file_torch
except ImportError:
    load_file_torch = None

from ..core.tensor import Tensor
from ..nn import Parameter
from ..models.llama import LlamaConfig

def load_config_from_hf(model_path: str) -> LlamaConfig:
    """Load LlamaConfig from config.json"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")
    
    with open(config_path, 'r') as f:
        hf_config = json.load(f)
        
    n_heads = hf_config.get("num_attention_heads", 32)
    n_kv_heads = hf_config.get("num_key_value_heads", n_heads)
    return LlamaConfig(
        hidden_size=hf_config.get("hidden_size", 4096),
        intermediate_size=hf_config.get("intermediate_size", 11008),
        n_layers=hf_config.get("num_hidden_layers", 32),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=hf_config.get("vocab_size", 32000),
        rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
        max_position_embeddings=hf_config.get("max_position_embeddings", 2048),
        max_batch_size=1
    )

def load_weights_from_hf(model: Any, model_path: str, device: str = "cuda"):
    """
    Load weights from HuggingFace safetensors files into NanoInfer model.
    
    Args:
        model: The NanoInfer LlamaModel instance.
        model_path: Path to the directory containing model files (config.json, *.safetensors).
        device: Device to load tensors onto ("cuda" or "cpu").
    """
    if load_file is None:
        raise ImportError("Please install safetensors: pip install safetensors")

    # 1. Load index if exists (for sharded models)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    weight_map = {}
    safetensors_files = []

    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        safetensors_files = sorted(list(set(weight_map.values())))
    else:
        # Check for single file
        single_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single_file):
            safetensors_files = ["model.safetensors"]
        else:
            # Maybe standard pytorch bin? (Not supported yet)
            raise FileNotFoundError(f"No model.safetensors or index found in {model_path}")

    # 2. Iterate over safetensors files
    print(f"Loading weights from {len(safetensors_files)} files...")
    
    # Mapping HF keys to NanoInfer keys
    # HF: model.layers.0.self_attn.q_proj.weight
    # Nano: layers[0].attention.q_proj.weight
    
    loaded_params = set()
    
    for sf in safetensors_files:
        file_path = os.path.join(model_path, sf)
        print(f"Processing {sf}...")
        try:
            state_dict = load_file(file_path)
        except Exception:
            if load_file_torch is None:
                raise
            state_dict = load_file_torch(file_path)
            state_dict = {k: v.detach().cpu().float().numpy() for k, v in state_dict.items()}

        for name, param in state_dict.items():
            # Skip if not weight (e.g. bias not used in some parts, or metadata)
            
            # Map name
            new_name = name
            
            # Standard mappings
            new_name = new_name.replace("self_attn", "attention")
            
            # Check if model is LlamaForCausalLM (has .model attribute)
            is_causal_lm = hasattr(model, 'model')
            
            if is_causal_lm:
                # LlamaForCausalLM structure:
                # .model -> LlamaModel
                # .lm_head -> Linear
                # HF keys: model.layers..., lm_head.weight
                # These map directly to LlamaForCausalLM attributes if we keep 'model.'
                pass
            else:
                # LlamaModel structure:
                # .layers...
                # HF keys: model.layers...
                # We need to strip 'model.' prefix
                if new_name.startswith("model."):
                    new_name = new_name[6:]
            
            # Assign to model
            try:
                _set_module_param(model, new_name, param, device)
                loaded_params.add(new_name)
            except AttributeError as e:
                # print(f"Warning: {e}")
                pass

    print(f"Successfully loaded {len(loaded_params)} parameters.")

def _set_module_param(model, name, value, device):
    """
    Recursively set parameter.
    name: layers.0.attention.q_proj.weight
    """
    parts = name.split('.')
    curr = model
    
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
             # Handle ModuleList (layers.0 -> layers[0])
             # The previous part must have been a list/container
             # Actually, in NanoInfer implementation:
             # self.layers is a ModuleList.
             # access pattern: model.layers[int(part)]
             pass
        else:
            if hasattr(curr, part):
                curr = getattr(curr, part)
                # Check if next part is digit (ModuleList access)
                if i + 1 < len(parts) - 1 and parts[i+1].isdigit():
                    idx = int(parts[i+1])
                    curr = curr[idx]
            else:
                # If we are here, maybe the mapping is slightly off or module doesn't exist
                # e.g. lm_head
                return

    # Last part is the attribute name (weight)
    param_name = parts[-1]
    
    if hasattr(curr, param_name):
        target_param = getattr(curr, param_name)
        if isinstance(target_param, (Parameter, Tensor)):
            # Convert numpy/tensor to NanoInfer Tensor
            # HF safetensors returns numpy arrays
            
            # Check shape
            if tuple(target_param.shape) != tuple(value.shape):
                print(f"Warning: Shape mismatch for {name}. Expected {target_param.shape}, got {value.shape}. Skipping.")
                return
                
            # Update data
            # We need to create a new Tensor on the target device with this data
            new_tensor = Tensor(value.astype(np.float32), device=device)
            
            if isinstance(target_param, Parameter):
                target_param.data = new_tensor.data # Update internal data pointer if possible, or replace
                # Since Parameter inherits from Tensor (usually) or holds a tensor. 
                # In NanoInfer:
                # class Parameter(Tensor): ...
                # So we can try to replace the object or update its contents.
                # Since Tensor is a wrapper around C++ object, we might need a way to load data.
                # Best way: setattr
                setattr(curr, param_name, Parameter(new_tensor, requires_grad=target_param.requires_grad))
            else:
                setattr(curr, param_name, new_tensor)
        else:
             print(f"Warning: Target {name} is not a Tensor/Parameter. It is {type(target_param)}")
