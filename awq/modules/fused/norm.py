import torch
from torch import nn
import awq_ext

class FasterTransformerRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        output = torch.empty_like(x)
        awq_ext.layernorm_forward_cuda(x, self.weight, output, self.variance_epsilon)
        return output 
