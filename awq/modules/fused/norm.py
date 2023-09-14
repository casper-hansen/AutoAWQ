import torch
from torch import nn
import awq_inference_engine

class FTLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        output = torch.empty_like(x)
        awq_inference_engine.layernorm_forward_cuda(x, self.weight, output, self.variance_epsilon)
        return output 
