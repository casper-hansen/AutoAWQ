import torch
import torch.nn as nn
from awq.utils.exllama_utils import unpack_reorder_pack

import exl_ext  # with CUDA kernels (AutoAWQ_kernels)


# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


class WQLinear_Exllama(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for Exllama kernels")

        self.q4 = None

        self.w_bit = w_bit
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size != -1 else in_features

        ##################################################################################
        ## These shapes are only for compatibility with the state_dict of WQLinear_GEMM ##
        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        ##################################################################################

        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
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

    def post_init(self):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        self.qweight, self.qzeros = unpack_reorder_pack(
            self.qweight, self.qzeros, self.w_bit
        )
        self.q4 = exl_ext.make_q4(
            self.qweight,
            self.qzeros,
            self.scales,
            none_tensor,  # g_idx
            self.qweight.device.index,  # device index
        )

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        raise NotImplementedError("Only inference is supported for Exllama kernels")

    def forward(self, x):
        assert self.q4 is not None, (
            "module.post_init() must be called before module.forward(). "
            "Use exllama_post_init() on the whole model."
        )

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
        exl_ext.q4_matmul(x, self.q4, out)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        if self.bias is not None:
            out.add_(self.bias)

        return out.view(out_shape)


def exllama_post_init(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, WQLinear_Exllama):
            submodule.post_init()

    return model
