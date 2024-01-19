import math
import torch
import torch.nn as nn
from awq.utils.exllama_utils import unpack_awq, pack_from_tensors

# adapted from https://github.com/AutoGPTQ/AutoGPTQ

try:
    from exllamav2_kernels import make_q_matrix, gemm_half_q_half
except ImportError as exllama_v2_import_exception:

    def error_raiser_exllama(*args, **kwargs):
        raise ValueError(
            f"Trying to use the exllama v2 backend, but could not import the C++/CUDA dependencies with the following error: {exllama_v2_import_exception}"
        )

    make_q_matrix = error_raiser_exllama
    gemm_half_q_half = error_raiser_exllama

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def _torch_device(idx):
    if idx == -1:
        return "cpu"
    return f"cuda:{idx}"


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


def ext_make_q_matrix(w: dict, temp_dq, key: str = None):
    """
    Create Q matrix
    """
    if w["scales"].dtype == torch.float:
        w["scales"] = w["scales"].half()

    return make_q_matrix(
        w["qweight"],
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        w["qzeros"],
        w["scales"],
        none_tensor,
        temp_dq,
    )


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
        self.maxq = 2**self.w_bit - 1
        self.in_features = in_features
        self.out_features = out_features
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

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None
        self.q_tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
            # "g_idx": self.g_idx,
        }
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq)

    @classmethod
    def from_wqlinear_gemm(cls, q_linear):
        # Create a new instance of the WQLinear class from ExllamaLinear with the same parameters
        exllama_linear = WQLinear_ExllamaV2(
            w_bit=q_linear.w_bit,
            group_size=q_linear.group_size,
            in_features=q_linear.in_features,
            out_features=q_linear.out_features,
            bias=q_linear.bias,
            dev=q_linear.qweight.device,
        )

        # Unpack the qweight and qzeros tensors
        fp16_weight, zeros = unpack_awq(
            q_linear.qweight,
            q_linear.qzeros,
            q_linear.scales,
            q_linear.w_bit,
            q_linear.group_size,
        )

        # Pack the qweight and qzeros tensors
        qweight, qzeros = pack_from_tensors(
            fp16_weight,
            zeros,
            q_linear.scales,
            q_linear.w_bit,
            q_linear.group_size,
        )

        # Copy the packed tensors to the ExllamaLinear instance
        exllama_linear.qweight.copy_(qweight)
        exllama_linear.qzeros.copy_(qzeros)
        exllama_linear.scales.copy_(q_linear.scales)

        return exllama_linear

    def forward(self, x, force_cuda=False):
        if x.dtype != torch.float16:
            x = x.half()

        output = ext_gemm_half_q_half(x, self.q_handle, self.out_features, force_cuda)

        if self.bias is not None:
            output.add_(self.bias)
        return output

    def temp_dq_size(self):
        return self.in_features * self.out_features * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.out_features * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)


class ExLlamaV2DeviceTensors:
    device_idx: int
    scratch_bytes: int
    scratch_idx: int
    scratch: torch.tensor = None

    def __init__(self, device_idx, scratch_bytes):
        self.device_idx = device_idx
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty(
            (self.scratch_bytes // 2,),
            dtype=torch.half,
            device=_torch_device(self.device_idx),
        )

    def get_scratch_slice(self, size_bytes):
        if self.scratch is None:
            self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
