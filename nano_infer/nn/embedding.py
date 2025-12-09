from .modules import Module
from .parameter import Parameter
from ..ops import functional as F
import numpy as np

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self.weight = Parameter(weight_data)

    def forward(self, input):
        return F.embedding(input, self.weight)