import torch
import torch.nn as nn

try:
    from intel_extension_for_pytorch.nn.modules.weight_only_quantization import WeightOnlyQuantizedLinear
    assert hasattr(WeightOnlyQuantizedLinear, "from_weight"), "The minimum version for ipex is at least 2.4"
    IPEX_INSTALLED = True
except:
    IPEX_INSTALLED = False


class WQLinear_IPEX(nn.Module):

    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        assert IPEX_INSTALLED, \
            "Please install IPEX package with `pip install intel_extension_for_pytorch`."
        assert w_bit == 4, "Only 4 bit are supported for now."

        self.use_bf16 = True # Intel platform support bf16 even without amx.

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.scale_dtype = torch.float32

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        self.pack_num = 32 // self.w_bit

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
        assert self.qweight.device.type == "cpu"
        self.ipex_linear = WeightOnlyQuantizedLinear.from_weight(self.qweight, self.scales, self.qzeros, \
                                                                self.in_features, self.out_features, None, self.bias, \
                                                                self.group_size, None, 0, 1)

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

    @torch.no_grad()
    def forward(self, x):
        assert IPEX_INSTALLED, (
            "IPEX kernels could not be loaded. "
            "Please install with `pip install intel_extension_for_pytorch` and "
            "refer to the detial https://github.com/intel/intel-extension-for-pytorch/tree/main")

        outputs = self.ipex_linear(x)

        return outputs

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
