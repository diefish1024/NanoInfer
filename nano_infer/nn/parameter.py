from ..core.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True):
        if isinstance(data, Tensor):
            super().__init__(data.data)
        else:
            super().__init__(data)
            
        self.requires_grad = requires_grad
    
    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"