import torch
import torch.nn as nn
import awq_inference_engine

class RMSNormInt8(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        output = torch.empty_like(x)
        awq_inference_engine.layernorm_forward_cuda(
            x, self.weight, output, self.variance_epsilon
        )
        out_int8 = output.round().clamp(-128, 127).to(torch.int8)
        return out_int8 
