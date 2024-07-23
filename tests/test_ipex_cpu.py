import torch
from awq.utils.packing_utils import dequantize_gemm
from intel_extension_for_pytorch.nn.modules.weight_only_quantization import WeightOnlyQuantizedLinear

assert hasattr(WeightOnlyQuantizedLinear, "from_weight"), "The minimum version for ipex is at least 2.4"
torch.manual_seed(0)

in_features = 256
out_features = 128
w_bit = 4
group_size = 32
torch_dtype = torch.bfloat16

MAX_INT32 = 0x7fffffff
MIN_INT32 = -MAX_INT32 - 1

qweight = torch.randint(
    MIN_INT32,
    MAX_INT32,
    (in_features, out_features // (32 // w_bit)),
    dtype=torch.int32,
    device="cpu",
)

qzeros = torch.randint(
    MIN_INT32,
    MAX_INT32,
    (in_features // group_size, out_features // (32 // w_bit)),
    dtype=torch.int32,
    device="cpu",
)

scales = torch.randn(
    (in_features // group_size, out_features),
    dtype=torch_dtype,
    device="cpu",
)

with torch.no_grad():
    fp_weight = dequantize_gemm(
        qweight,
        qzeros,
        scales,
        w_bit,
        group_size
    )
    
    ipex_linear = WeightOnlyQuantizedLinear.from_weight(qweight, scales, qzeros, \
                                                        in_features, out_features, None, None, \
                                                        group_size, None, 0, 1)


    input = torch.rand(1, in_features, dtype=torch_dtype)
    torch_out = torch.matmul(input, fp_weight)

    ipex_dst = ipex_linear(input)
    results = torch.amax(ipex_dst - torch_out)

    assert(torch.allclose(ipex_dst, torch_out, rtol=0.06))