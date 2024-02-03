import torch
from torch import nn

try:
    import awq_ext  # with CUDA kernels

    AWQ_INSTALLED = True
except:
    AWQ_INSTALLED = False


class FasterTransformerRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        assert AWQ_INSTALLED, (
            "AWQ kernels could not be loaded. "
            "Please install them from https://github.com/casper-hansen/AutoAWQ_kernels"
        )

        output = torch.empty_like(x)
        awq_ext.layernorm_forward_cuda(x, self.weight, output, self.variance_epsilon)

        return output
