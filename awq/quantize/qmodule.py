import math
import torch
import torch.nn as nn
import awq_inference_engine
from exllama_kernels import make_q4, q4_matmul

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")

def ext_make_q4(qweight, qzeros, scales, g_idx, device):
    """Construct Q4Matrix, return handle"""
    if not qweight.shape[1] == qzeros.shape[1] * 8:
        raise Exception(f"qweight.shape[1] ({qweight.shape[1]}) "
                        "must have the same shape as qzeros.shape[1]*8 ({qzeros.shape[1]*8})")

    return make_q4(qweight,
                   qzeros,
                   scales,
                   g_idx if g_idx is not None else none_tensor,
                   device)

def ext_q4_matmul(x, q4, q4_width):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)

    q4_matmul(x, q4, output)

    return output.view(outshape)

class ExllamaLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, qweight:torch.Tensor, qzeros:torch.Tensor, 
                       scales:torch.Tensor, bias:torch.Tensor, q_config:dict):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.width = qweight.shape[1]
        self.w_bit = q_config["w_bit"]
        self.group_size = q_config["q_group_size"]

        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features // 32 * q_config["w_bit"]), dtype=torch.int32)
        )
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(in_features / q_config["q_group_size"]), out_features // 32 * q_config["w_bit"]), dtype=torch.int32)
        )
        self.register_buffer(
            'scales',
            torch.zeros((math.ceil(in_features / q_config["q_group_size"]), out_features), dtype=torch.float16)
        )

        self.q4 = ext_make_q4(
            qweight.transpose(0,1),
            qzeros,
            scales,
            None,
            0 # device index
        )

    def forward(self, x):
        out = ext_q4_matmul(x.half(), self.q4, self.width)

        if self.bias is not None:
            out.add_(self.bias)
        
        return out

class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)
    
    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer('qweight', torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales
        
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[idx // group_size]) / awq_linear.scales[idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=intweight.device)           
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=zeros.device)
        
        for col in range(zeros.shape[1] // pack_num):     
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        out = awq_inference_engine.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )
