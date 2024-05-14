import torch
from awq.utils.packing_utils import unpack_awq, reverse_awq_order
from awq.modules.linear.gemm_qbits import BITS_DTYPE_MAPPING, convert_dtype_torch2str
from awq.utils.packing_utils import dequantize_gemm
from intel_extension_for_transformers import qbits
torch.manual_seed(0)

in_features = 256
out_features = 128
w_bit = 4
group_size = 32
torch_dtype = torch.bfloat16 if qbits.check_isa_supported("AMX") else torch.float32

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
    intweight, zeros = unpack_awq(qweight, qzeros, w_bit) # weight: k x n zeros: k / group_size x n
    intweight, zeros = reverse_awq_order(intweight, zeros, w_bit) # weight: k x n zeros: k / group_size x n
    # overflow checks
    intweight = torch.bitwise_and(intweight, (2**w_bit) - 1) - (2**(w_bit - 1))
    zeros = torch.bitwise_and(zeros, (2**w_bit) - 1) - (2**(w_bit - 1))
    g_idx = torch.empty(0, dtype=torch.int32)
    qbits_qweight = qbits.repack_quantized_weight(intweight, scales.float().contiguous(), zeros, g_idx,
                                                  BITS_DTYPE_MAPPING[w_bit],
                                                  "fp32",
                                                  convert_dtype_torch2str(torch_dtype),
                                                  True,
                                                  group_size)
    qbits_out = torch.zeros(in_features, out_features, dtype=torch.float32)
    qbits.dequantize_packed_weight(
        qbits_qweight, qbits_out, False, convert_dtype_torch2str(torch_dtype), BITS_DTYPE_MAPPING[w_bit], "fp32")
    qbits_out = qbits_out.to(torch_dtype)
    assert(torch.allclose(qbits_out, fp_weight, rtol=0.0001))

    input = torch.rand(1, in_features, dtype=torch_dtype)
    torch_out = torch.matmul(input, fp_weight)

    qbits_dst = torch.zeros(1, out_features, dtype=torch.bfloat16)
    qbits.woq_linear(
        input, qbits_qweight, torch.empty(0), qbits_dst, convert_dtype_torch2str(torch_dtype), "int4_clip", "fp32", True)
    results = torch.amax(qbits_dst - torch_out)

    assert(torch.allclose(qbits_dst, torch_out, rtol=0.03))