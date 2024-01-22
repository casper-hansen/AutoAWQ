import torch

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

import awq_ext
from awq.utils.packing_utils import dequantize_gemm

in_features = 4096
out_features = 1792
w_bit = 4
group_size = 128

MAX_INT32 = 0x7fffffff
MIN_INT32 = -MAX_INT32 - 1

qweight = torch.randint(
    MIN_INT32,
    MAX_INT32,
    (in_features, out_features // (32 // w_bit)),
    dtype=torch.int32,
    device="cuda",
)

qzeros = torch.randint(
    MIN_INT32,
    MAX_INT32,
    (in_features // group_size, out_features // (32 // w_bit)),
    dtype=torch.int32,
    device="cuda",
)

scales = torch.randn(
    (in_features // group_size, out_features),
    dtype=torch.float16,
    device="cuda",
)

with torch.no_grad():
    cuda_out = awq_ext.dequantize_weights_cuda(
        qweight,
        scales,
        qzeros,
        0,
        0,
        0,
        False
    )
    torch_out = dequantize_gemm(
        qweight,
        qzeros,
        scales,
        w_bit,
        group_size
    )

assert(torch.allclose(cuda_out, torch_out, rtol=0.0001))