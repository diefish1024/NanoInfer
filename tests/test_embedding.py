import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
import nano_infer
from nano_infer.core.tensor import Tensor
from nano_infer.nn.embedding import Embedding
import nano_infer.ops.functional as F

def test_embedding_correctness():
    Vocab = 1000
    Dim = 128
    Batch = 2
    Seq = 5
    
    torch_emb = torch.nn.Embedding(Vocab, Dim).cuda()
    nano_emb = Embedding(Vocab, Dim)
    nano_emb.weight.data = Tensor(torch_emb.weight.detach().cpu().numpy()).to_cuda().data

    indices_np = np.random.randint(0, Vocab, (Batch, Seq)).astype(np.int32)
    
    indices_pt = torch.tensor(indices_np).cuda()
    indices_nano = Tensor(indices_np).to_cuda()
    
    y_pt = torch_emb(indices_pt)
    y_nano = nano_emb(indices_nano)
    
    diff = np.abs(y_pt.detach().cpu().numpy() - y_nano.numpy()).max()
    print(f"Max Difference: {diff}")
    if diff < 1e-5:
        print("✅ Embedding Test Passed!")
    else:
        print("❌ Embedding Test Failed!")


if __name__ == "__main__":
    test_embedding_correctness()