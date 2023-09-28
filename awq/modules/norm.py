import torch
import torch.nn as nn
import awq_inference_engine

class RMSNormInt8(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32), requires_grad=False)
        self.weight.data = self.weight.data.to(torch.int8)
        self.variance_epsilon = torch.tensor(eps, dtype=torch.float32).half()

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = x.to(torch.int8)
        output = torch.empty_like(x, dtype=torch.int8)
        print(x.dtype, self.weight.dtype, output.dtype, self.variance_epsilon.dtype)

        awq_inference_engine.layernorm_forward_cuda(
            x, self.weight, output, self.variance_epsilon
        )
        return output
