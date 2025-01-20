import torch
import torch.nn as nn
from .gemm import WQLinear_GEMM
from awq.utils.packing_utils import dequantize_gemm

try:
    from intel_extension_for_pytorch.llm.quantization import IPEXWeightOnlyQuantizedLinear
    assert hasattr(IPEXWeightOnlyQuantizedLinear, "from_weight"), "The minimum version for ipex is at least 2.4"
    IPEX_INSTALLED = True
except:
    IPEX_INSTALLED = False


class WQLinear_IPEX(WQLinear_GEMM):

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, training=False):
        nn.Module.__init__(self)
        assert IPEX_INSTALLED, \
            "Please install IPEX package with `pip install intel_extension_for_pytorch`."
        assert w_bit == 4, "Only 4 bit are supported for now."

        self.use_bf16 = True # Intel platform support bf16 even without amx.

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.scale_dtype = torch.float32
        self.training = training

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        self.pack_num = 32 // self.w_bit

        self.init_ipex = False

        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // self.pack_num),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
                device=dev,
            ))
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features), dtype=torch.bfloat16 if self.use_bf16 else torch.float32, device=dev),
            )
        else:
            self.register_buffer(
                "bias",
                None,
            )
        qweight = torch.zeros((in_features, out_features // self.pack_num), dtype=torch.int32, device=dev)
        self.register_buffer("qweight", qweight)

    def post_init(self):
        device_type = self.qweight.device.type
        if device_type != "meta":
            assert device_type in ("cpu", "xpu")

    def init_ipex_linear(self):
        if not self.training:
            self.ipex_linear = IPEXWeightOnlyQuantizedLinear.from_weight(self.qweight, self.scales, self.qzeros, \
                                                                    self.in_features, self.out_features, None, self.bias, \
                                                                    self.group_size, None, quant_method=1, dtype=0)

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None):
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

        raise NotImplementedError("Only inference is supported for IPEX kernels")

    def forward(self, x):
        assert IPEX_INSTALLED, (
            "IPEX kernels could not be loaded. "
            "Please install with `pip install intel_extension_for_pytorch` and "
            "refer to the detial https://github.com/intel/intel-extension-for-pytorch/tree/main")

        if not self.init_ipex:
            self.init_ipex_linear()
            self.init_ipex = True

        if hasattr(self, "ipex_linear"):
            with torch.no_grad():
                outputs = self.ipex_linear(x)
        else:
            outputs = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.w_bit, self.group_size).to(x.dtype)
            outputs = torch.matmul(x, outputs)

        return outputs
    
    def backward(self, grad_output):
        weights = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.w_bit, self.group_size).to(grad_output.dtype)
        batch_size = grad_output.shape[0]
        grad_input = grad_output.bmm(weights.transpose(0, 1).unsqueeze(0).repeat(batch_size, 1, 1))

        return grad_input, None, None, None, None, None, None, None

    def extra_repr(self) -> str:
        return ("in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.w_bit,
            self.group_size,
        ))


def ipex_post_init(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, WQLinear_IPEX):
            submodule.post_init()

    return model
