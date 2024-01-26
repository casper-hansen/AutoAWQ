import torch
import torch.nn as nn
from typing import Dict
from awq.utils.packing_utils import unpack_reorder_pack

try:
    import exlv2_ext  # with CUDA kernels (AutoAWQ_kernels)

    EXLV2_INSTALLED = True
except:
    EXLV2_INSTALLED = False

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


class WQLinear_ExllamaV2(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.q_handle = None

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

    def post_init(self, scratch_space: "ScratchSpace"):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        self.qweight, self.qzeros = unpack_reorder_pack(
            self.qweight, self.qzeros, self.w_bit
        )

        temp_dq_size = self.temp_dq_size()
        temp_dq = scratch_space.get_slice(temp_dq_size)
        self.q_handle = exlv2_ext.make_q_matrix(
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

        raise NotImplementedError("Only inference is supported for ExllamaV2 kernels")

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

    def forward(self, x):
        assert self.q_handle is not None, (
            "module.post_init() must be called before module.forward(). "
            "Use exllamav2_post_init() on the whole model."
        )
        assert EXLV2_INSTALLED, (
            "ExllamaV2 kernels are not installed. "
            "Please install AWQ compatible ExllamaV2 kernels from AutoAWQ_kernels."
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
        exlv2_ext.gemm_half_q_half(x, self.q_handle, out, False)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        if self.bias is not None:
            out.add_(self.bias)

        return out.view(out_shape)


class ScratchSpace:
    def __init__(self, scratch_bytes, dev):
        self.scratch_bytes = scratch_bytes
        self.scratch = torch.empty(
            self.scratch_bytes // 2,
            dtype=torch.float16,
            device=dev,
        )

    def get_slice(self, size_bytes):
        size_halfs = next_multiple(size_bytes, 128) // 2
        scratch_slice = self.scratch.narrow(0, 0, size_halfs)

        return scratch_slice


def exllamav2_post_init(model, max_input_len: int = 2048, max_batch_size: int = 8):
    # we search for the maximum number of bytes required for each device's scratch space
    fixed_bytes: Dict[torch.device, int] = {}
    for _, submodule in model.named_modules():
        if isinstance(submodule, WQLinear_ExllamaV2):
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed(
                max_input_len=max_input_len, max_batch_size=max_batch_size
            )
            fixed_bytes[device] = max(fixed_bytes.get(device, 0), scratch_fixed)

    # we allocate a model-persistent scratch space for each device
    model.scratch_spaces: Dict[torch.device, ScratchSpace] = {}
    for device, scratch_bytes in fixed_bytes.items():
        model.scratch_spaces[device] = ScratchSpace(scratch_bytes, device)

    for _, submodule in model.named_modules():
        if isinstance(submodule, WQLinear_ExllamaV2):
            device = submodule.qweight.device
            submodule.post_init(scratch_space=model.scratch_spaces[device])

    return model


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple
