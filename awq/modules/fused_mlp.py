import torch
import torch.nn as nn
import awq_inference_engine
import torch.nn.functional as F

class QuantMPTMLP(nn.Module):
    def __init__(
        self,
        up_proj,
        act,
        down_proj
    ):
        super().__init__()
        self.register_buffer('up_proj_qweight', up_proj.qweight)
        self.register_buffer('up_proj_scales', up_proj.scales)
        self.register_buffer('up_proj_qzeros', up_proj.qzeros)

        self.up_proj = up_proj
        self.act = act
        self.down_proj = down_proj
    
    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        x = awq_inference_engine.gemm_forward_cuda(x, self.up_proj_qweight, self.up_proj_scales, self.up_proj_qzeros, 8)

        return self.down_proj(self.act(x))

class QuantLlamaMLP(nn.Module):

    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj,
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

    def forward(self, x):
        return self.down_proj(self.our_llama_mlp(x))
    
    def our_llama_mlp(self, x):
        out_shape = x.shape[:-1] + (self.intermediate_size, )
        x = x.reshape(-1, x.shape[-1])
        gate_output = awq_inference_engine.gemm_forward_cuda(
            x, self.gate_proj_qweight, self.gate_proj_scales, self.gate_proj_qzeros, 8
        )
        gate_output = F.silu(gate_output)
        up_output = awq_inference_engine.gemm_forward_cuda(
            x, self.up_proj_qweight, self.up_proj_scales, self.up_proj_qzeros, 8
        )
        c = gate_output * up_output
        c = c.reshape(out_shape)
        return c
