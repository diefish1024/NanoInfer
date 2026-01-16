from collections import OrderedDict
from .parameter import Parameter
from ..core.tensor import Tensor
from ..ops import functional as F
import numpy as np

class Module:
    def __init__(self):
        self._modules = OrderedDict()
        self._parameters = OrderedDict()
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def state_dict(self, prefix='', destination=None):
        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(prefix + name + '.', destination)

        return destination

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"[Warning] Unexpected key: {name}")
                continue
            
            # assume shape is same
            if isinstance(param, Tensor):
                # TODO: copy
                own_state[name].data = param.data 
            else: # numpy or list
                new_tensor = Tensor(param)
                own_state[name].data = new_tensor.data
        
        print(f"Loaded {len(state_dict)} parameters successfully.")

    def to(self, device):
        for param in self._parameters.values():
            if device == 'cuda':
                param.to_cuda()
            else:
                param.to_cpu()
        
        for module in self._modules.values():
            module.to(device)
            
        return self

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __getitem__(self, idx):
        # Allow indexing like a normal list
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module):
        # Use the index as the module name for registration
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        for module in modules:
            self.append(module)
        return self

    def add_module(self, name, module):
        # Manually register into the _modules OrderedDict
        self._modules[name] = module

    def forward(self, *args, **kwargs):
        # ModuleList itself doesn't have a forward pass logic
        raise NotImplementedError