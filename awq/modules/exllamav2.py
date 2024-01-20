import math
import torch
import torch.nn as nn
from awq.utils.exllama_utils import unpack, pack, awq_reverse_reorder, none_tensor

import exllamav2_kernels  # with CUDA kernels (AutoAWQ_kernels)


class WQLinear_ExllamaV2(nn.Module):
    QUANT_TYPE: str = "exllamav2"

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.q_handle = None
        self.q_tensors = None
        self.padding = -out_features % 32

        self.w_bit = w_bit
        self.in_features = in_features
        self.out_features = out_features + self.padding
        self.group_size = group_size if group_size != -1 else in_features

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features // 32 * w_bit),
                out_features,
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(in_features / group_size),
                    out_features // 32 * w_bit,
                ),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(in_features / group_size), out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    def temp_dq_size(self):
        """
        Returns the size of the temporary buffer required for the dq kernel.
        """
        return self.in_features * self.out_features * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        """
        Returns the size of the temporary buffer required for the fwd kernel.
        """
        return self.out_features * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        """
        Returns the size of the fixed scratch space required for the kernel.
        """
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_handle = exllamav2_kernels.make_q_matrix(
            self.qweight,
            none_tensor,
            none_tensor,
            none_tensor,
            none_tensor,
            none_tensor,
            self.qzeros,
            self.scales,
            none_tensor,
            temp_dq,
        )

    @classmethod
    def from_wqlinear_gemm(cls, q_linear):
        exllama_linear = WQLinear_ExllamaV2(
            w_bit=q_linear.w_bit,
            group_size=q_linear.group_size,
            in_features=q_linear.in_features,
            out_features=q_linear.out_features,
            dev=q_linear.qweight.device,
            bias=q_linear.bias,
        )

        # Create a new instance of the WQLinear class from ExllamaLinear with the same parameters
        bits = q_linear.w_bit
        qzeros = q_linear.qzeros
        qweight = q_linear.qweight

        # Unpack the qweight and qzeros tensors
        iweight, izeros = unpack(qweight, qzeros, bits)
        # Reverse reorder the iweight and izeros tensors
        iweight, izeros = awq_reverse_reorder(iweight, izeros, bits)
        # Subtract 1 from the izeros tensor
        izeros = torch.bitwise_and(izeros - 1, (2**bits) - 1)
        # Pack the qweight and qzeros tensors
        qweight, qzeros = pack(iweight, izeros, bits)

        # Copy the packed tensors to the ExllamaLinear instance
        exllama_linear.scales.copy_(q_linear.scales)
        exllama_linear.qweight.copy_(qweight)
        exllama_linear.qzeros.copy_(qzeros)

        return exllama_linear

    def forward(self, x):
        input_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.out_features,)

        if input_dtype != torch.float16:
            x = x.to(dtype=torch.float16)

        x = x.view(-1, x.shape[-1])

        out = torch.empty(
            (x.shape[0], self.out_features),
            dtype=torch.float16,
            device=x.device,
        )
        exllamav2_kernels.gemm_half_q_half(x, self.q_handle, out, False)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        if self.bias is not None:
            out.add_(self.bias)

        return out.view(out_shape)
