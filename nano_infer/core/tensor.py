import numpy as np
from ..backend import _nano_infer

CppTensor = _nano_infer.Tensor
Device = _nano_infer.Device
DType = _nano_infer.DType

class Tensor:
    def __init__(self, data, dtype=None, device=None):
        if isinstance(data, CppTensor):
            self.data = data
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            if dtype is None:
                if np.issubdtype(data.dtype, np.integer):
                    data = data.astype(np.int32)
                elif np.issubdtype(data.dtype, np.floating):
                    data = data.astype(np.float32)
                else:
                    data = data.astype(np.float32)
            else:
                data = data.astype(dtype)

            if device is None:
                target_device = Device.CPU
            elif isinstance(device, str):
                if device.lower() == 'cuda':
                    target_device = Device.CUDA
                elif device.lower() == 'cpu':
                    target_device = Device.CPU
                else:
                    raise ValueError(f"Unknown device string: {device}")
            else:
                target_device = device
            
            self.data = CppTensor(data, target_device)

    @staticmethod
    def empty(shape, dtype=DType.Float32, device=None):
        target_device = Device.CPU
        if device is None:
            target_device = Device.CPU
        elif isinstance(device, str):
            target_device = Device.CUDA if device.lower() == 'cuda' else Device.CPU
        elif isinstance(device, Device):
            target_device = device

        target_dtype = dtype
        if target_dtype == np.float32:
            target_dtype = DType.Float32
        elif target_dtype == np.int32:
            target_dtype = DType.Int32

        cpp_tensor = CppTensor(list(shape), target_dtype, target_device)
        return Tensor(cpp_tensor)
    
    def numpy(self):
        if self.data.device == Device.CUDA:
            cpu_tensor = self.data.to_cpu()
            return np.array(cpu_tensor, copy=True)
        else:
            return np.array(self.data, copy=True)
        
    @property
    def dtype(self):
        if self.data.dtype == DType.Int32:
            return np.int32
        return np.float32
    
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
        typestr = "<f4"
        if self.data.dtype == DType.Int32:
            typestr = "<i4"
        return {
            "shape": self.shape,
            "typestr": typestr,
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

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            new_shape = list(shape[0])
        else:
            new_shape = list(shape)

        if -1 in new_shape:
            num_elements = self.numel
            known_elements = 1
            minus_one_idx = -1
            for i, dim in enumerate(new_shape):
                if dim == -1:
                    minus_one_idx = i
                else:
                    known_elements *= dim
            new_shape[minus_one_idx] = num_elements // known_elements

        self.data.reshape(new_shape)
        return self

    def contiguous(self):
        return self

    def __repr__(self):
        if self.data.device == Device.CUDA:
             d_str = "int32" if self.data.dtype == DType.Int32 else "float32"
             return f"Tensor({self.shape}, device='cuda', dtype={d_str})"
        return f"Tensor(data={self.numpy()}, device='cpu')"
    
    def __add__(self, other):
        from ..ops import functional as F
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.add(self, other)
    
    def __mul__(self, other):
        from ..ops import functional as F
        if isinstance(other, (int, float)):
            out_cpp = self.data.mul_scalar(float(other))
            return Tensor(out_cpp, device=self.device)
        other = other if isinstance(other, Tensor) else Tensor(other)
        return F.mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            out_cpp = self.data.mul_scalar(1.0 / float(other))
            return Tensor(out_cpp, device=self.device)
        raise TypeError(f"Unsupported operand type(s) for /: 'Tensor' and '{type(other)}")

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

    def slice(self, dim, start, end):
        from ..ops import functional as F
        return F.slice(self, dim, start, end)

    def transpose(self, dim0, dim1):
        from ..ops import functional as F
        return F.transpose(self, dim0, dim1)
