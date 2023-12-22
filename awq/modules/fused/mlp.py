import torch.nn as nn
import awq_ext
import torch.nn.functional as F
from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV


class QuantFusedMLP(nn.Module):
    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj,
        activation = F.silu,
    ):
        super().__init__()

        self.register_buffer('gate_proj_qweight', gate_proj.qweight)
        self.register_buffer('gate_proj_scales', gate_proj.scales)
        self.register_buffer('gate_proj_qzeros', gate_proj.qzeros)
        self.register_buffer('up_proj_qweight', up_proj.qweight)
        self.register_buffer('up_proj_scales', up_proj.scales)
        self.register_buffer('up_proj_qzeros', up_proj.qzeros)

        self.in_features = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features
        self.out_features = down_proj.out_features
        self.w_bit = gate_proj.w_bit
        self.down_proj = down_proj

        if isinstance(down_proj, WQLinear_GEMV):
            self.linear = awq_ext.gemv_forward_cuda
            self.group_size = down_proj.group_size
        else:
            self.linear = awq_ext.gemm_forward_cuda
            self.group_size = 8

        self.activation = activation

    def forward(self, x, routing_weights=None):
        out_shape = x.shape[:-1] + (self.intermediate_size,)
        x = x.reshape(-1, x.shape[-1])
        gate_output = self.linear(
            x,
            self.gate_proj_qweight,
            self.gate_proj_scales,
            self.gate_proj_qzeros,
            self.group_size,
        )
        up_output = self.linear(
            x,
            self.up_proj_qweight,
            self.up_proj_scales,
            self.up_proj_qzeros,
            self.group_size,
        )
        x = self.activation(gate_output) * up_output
        x = x.reshape(out_shape)
        x = self.down_proj(x)

        if routing_weights is not None:
            x = routing_weights * x

        return x
        

class QuantLlamaMLP(QuantFusedMLP):
    r"""
    QuantLlamaMLP class kept for backward compatibilty, in the future, users 
    should always use `QuantFusedMLP` class instead.
    """
    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj
    ):
        super().__init__(gate_proj, down_proj, up_proj)