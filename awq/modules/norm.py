import torch
import torch.nn as nn
import awq_inference_engine

class RMSNormInt8(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        output = torch.empty_like(x, dtype=torch.int8)
        awq_inference_engine.layernorm_forward_cuda(
            x, self.weight, output, self.variance_epsilon
        )
        return output
