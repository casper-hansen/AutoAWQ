import math
import torch
import torch.nn as nn
from awq.utils.exllama_utils import unpack_awq, pack_from_tensors

try:
    from exllama_kernels import make_q4, q4_matmul
except ImportError as exllama_import_exception:

    def error_raiser_exllama(*args, **kwargs):
        raise ValueError(
            f"Trying to use the exllama backend, but could not import the C++/CUDA dependencies with the following error: {exllama_import_exception}"
        )

    make_q4 = error_raiser_exllama
    q4_matmul = error_raiser_exllama

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def ext_make_q4(qweight, qzeros, scales, g_idx, device):
    """Construct Q4Matrix, return handle"""
    return make_q4(
        qweight, qzeros, scales, g_idx if g_idx is not None else none_tensor, device
    )


def ext_q4_matmul(x, q4, q4_width):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)

    q4_matmul(x, q4, output)

    return output.view(outshape)


class WQLinear_Exllama(nn.Module):
    QUANT_TYPE: str = "exllama"

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features // 32 * w_bit), out_features, dtype=torch.int32, device=dev
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
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None

    def post_init(self):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        self.q4 = ext_make_q4(
            self.qweight,
            self.qzeros,
            self.scales,
            None,
            self.qweight.device.index,  # device index
        )

    @classmethod
    def from_wqlinear_gemm(cls, q_linear):
        # Create a new instance of the WQLinear class from ExllamaLinear with the same parameters
        exllama_linear = WQLinear_Exllama(
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

    def forward(self, x):
        out = ext_q4_matmul(x.half(), self.q4, self.out_features)

        if self.bias is not None:
            out.add_(self.bias)

        return out
