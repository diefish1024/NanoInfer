import numpy as np
from ..backend import _nano_infer

CppTensor = _nano_infer.Tensor
Device = _nano_infer.Device

class Tensor:
    def __init__(self, data, device=None):
        if isinstance(data, CppTensor):
            self.data = data
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            
            self.data = CppTensor(data.shape, Device.CPU)
            
            dst = np.array(self.data, copy=False)
            dst[:] = data

            if device != 'cpu': 
                self.data.to_cuda()
    
    def numpy(self):
        if self.data.device == Device.CUDA:
            cpu_tensor = self.data.to_cpu()
            return np.array(cpu_tensor, copy=True)
        else:
            return np.array(self.data, copy=True)
    
    @property
    def strides(self):
        return tuple(self.data.strides)

    @property
    def data_ptr(self): # for triton
        return self.data.data_ptr

    @property
    def numel(self):
        return self.data.size
    
    @property
    def __cuda_array_interface__(self):
        if self.data.device != Device.CUDA:
            raise AttributeError("Tensor is on CPU, no CUDA interface available.")
        
        return {
            "shape": self.shape,
            "typestr": "<f4", # float32, little-endian
            "data": (self.data_ptr, False), # (ptr, read_only_flag)
            "version": 3,
            "strides": None
        }
    
    @property
    def device(self):
        return self.data.device

    def to_cuda(self):
        self.data.to_cuda()
        return self

    def to_cpu(self):
        self.data.to_cpu()
        return self

    def to(self, device):
        if device == Device.CUDA:
            return self.to_cuda()
        elif device == Device.CPU:
            return self.to_cpu()
        return self
    
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        if self.data.device == Device.CUDA:
             return f"Tensor({self.shape}, device='cuda')"
        return f"Tensor(data={self.numpy()}, device='cpu')"
    
    def __add__(self, other):
        from ..ops import functional as F
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.add(self, other)
    
    def __mul__(self, other):
        from ..ops import functional as F
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.mul(self, other)

    def __matmul__(self, other):
        from ..ops import functional as F
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.matmul(self, other)
    
    def add(self, other):
        return self.__add__(other)
        
    def mul(self, other):
        return self.__mul__(other)
        
    def matmul(self, other, trans_a=False, trans_b=False):
        from ..ops import functional as F
        other = other if isinstance(other, Tensor) else Tensor(other)
        return  F.matmul(self, other, trans_a, trans_b)