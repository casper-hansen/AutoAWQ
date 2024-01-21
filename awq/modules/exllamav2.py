import torch
import torch.nn as nn
from awq.utils.exllama_utils import unpack_reorder_pack

import exllamav2_kernels  # with CUDA kernels (AutoAWQ_kernels)

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


class WQLinear_ExllamaV2(nn.Module):
    QUANT_TYPE: str = "exllamav2"

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.q_handle = None
        self.q_tensors = None

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
        ## These shapes are only for compatibility with the state_dict of WQLinear_GEMM ##
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

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        self.qweight, self.qzeros = unpack_reorder_pack(
            self.qweight, self.qzeros, self.w_bit
        )

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


def exllamav2_post_init(model):
    fixed_bytes = {}
    for _, submodule in model.named_modules():
        if isinstance(submodule, WQLinear_ExllamaV2):
            device = submodule.qweight.device
            if device not in fixed_bytes:
                fixed_bytes[device] = 0

            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(fixed_bytes[device], scratch_fixed)

    device_tensors = {}
    for device, scratch_bytes in fixed_bytes.items():
        device_tensors[device] = ExLlamaV2DeviceTensors(device, scratch_bytes)

    # have persistent buffers, otherwise we will get OOM
    model.device_tensors = device_tensors

    for _, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
            device = submodule.qweight.device
            submodule.post_init(temp_dq=model.device_tensors[device])

    return model


class ExLlamaV2DeviceTensors:
    dev: torch.device
    scratch_bytes: int
    scratch: torch.tensor = None

    def __init__(self, dev, scratch_bytes):
        self.dev = dev
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty(
            self.scratch_bytes // 2,
            dtype=torch.float16,
            device=self.dev,
        )

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
