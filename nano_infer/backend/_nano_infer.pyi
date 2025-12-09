from typing import List, ClassVar

class Device:
    """
    Device Enum for Tensor placement.
    """
    CPU: ClassVar['Device']
    CUDA: ClassVar['Device']
    
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __init__(self, value: int) -> None: ...

class Tensor:
    """
    NanoInfer C++ Tensor Binding.
    """
    # --- Properties (Read-only attributes) ---
    shape: List[int]
    strides: List[int]
    size: int
    device: Device
    data_ptr: int

    # --- Constructor ---
    def __init__(self, shape: List[int], device: Device = Device.CUDA) -> None: ...

    # --- Data Movement ---
    def to_cuda(self) -> 'Tensor': ...
    def to_cpu(self) -> 'Tensor': ...

    # --- Operators ---
    def add(self, other: 'Tensor') -> 'Tensor': ...
    def mul(self, other: 'Tensor') -> 'Tensor': ...
    
    def matmul(self, other: 'Tensor', trans_a=False, trans_b=False) -> 'Tensor': ...

    def silu(self, out: 'Tensor') -> 'Tensor': ...

    def embedding(input: 'Tensor', weight: 'Tensor', out: 'Tensor') -> 'Tensor': ...

    # --- Special Methods ---
    def __repr__(self) -> str: ...