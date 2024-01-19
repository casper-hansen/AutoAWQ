import numpy as np
import torch

# copied from https://github.com/AutoGPTQ/AutoGPTQ/pull/484


def awq_reverse_reorder_int_tensor(int_tensor, bits: int):
    assert bits == 4

    int_tensor = int_tensor.T.contiguous()
    compress_ratio = 32 // bits
    assert int_tensor.shape[-1] % compress_ratio == 0

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    order_tensor = torch.tensor(
        order_map, dtype=torch.int32, device=int_tensor.device
    ).reshape(1, -1)
    order_tensor = order_tensor.repeat(int_tensor.shape[1] // compress_ratio, 1)
    order_tensor = order_tensor + torch.arange(
        0,
        int_tensor.shape[1],
        compress_ratio,
        dtype=torch.int32,
        device=int_tensor.device,
    ).reshape(-1, 1)
    order_tensor = order_tensor.reshape(-1)

    reverse_order_tensor = torch.arange(order_tensor.shape[0]).cuda()[order_tensor]
    reverse_order_tensor = reverse_order_tensor[order_tensor]
    int_tensor = int_tensor[:, reverse_order_tensor]
    return int_tensor


def unpack_awq(
    awq_qweight: torch.Tensor,
    awq_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    """
    Args:
        awq_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features // (32 // bits))
        awq_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features // (32 // bits))
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        fp16_weight (`torch.LongTensor`):
            With shape (in_features, out_features).
        zeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features).
    """
    assert bits == 4

    qzeros = awq_qzeros.cuda()
    qweight = awq_qweight.cuda()
    qweight = qweight.T.contiguous()

    scales = awq_scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    infeatures = awq_qweight.shape[0]

    wf = torch.tensor(
        list(range(0, 32, bits)), dtype=torch.int32, device=qzeros.device
    ).unsqueeze(0)
    zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
        torch.int16 if bits == 8 else torch.int8
    )

    # zeros = zeros + 1

    torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)
    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)
    weight = weight.reshape(-1, group_size, weight.shape[2])

    weight = weight.view(-1, weight.shape[-1])
    zeros = zeros.view(-1, zeros.shape[-1])

    zeros = zeros.T.contiguous()
    zeros = awq_reverse_reorder_int_tensor(zeros, bits)
    weight = awq_reverse_reorder_int_tensor(weight, bits)

    # Dequantize weights.
    scales = awq_scales.cuda()
    zeros = zeros.contiguous()
    scale_zeros = zeros * scales

    g_idx = torch.tensor(
        [i // group_size for i in range(infeatures)], dtype=torch.int32
    )
    scale_mat = scales[g_idx]
    scale_zeros_mat = scale_zeros[g_idx].half()

    qdq_weight_T = weight * scale_mat - scale_zeros_mat.half()

    fp16_weight = qdq_weight_T.T.cuda()

    return fp16_weight, zeros


def pack_from_tensors(
    unpacked_qweight: torch.Tensor,
    unpacked_qzeros: torch.Tensor,
    awq_scales: torch.Tensor,
    bits: int,
    group_size: int,
):
    """
    Args:
        unpacked_qweight (`torch.LongTensor`):
            Expected shape: (in_features, out_features)
        unpacked_qzeros (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)
        awq_scales (`torch.LongTensor`):
            Expected shape: (in_features // group_size, out_features)

    Returns:
        qweight (`torch.LongTensor`):
            With shape (in_features // (32 // bits), out_features)
        qzeros (`torch.LongTensor`):
            With shape (in_features // group_size, out_features // (32 // bits))
    """
    assert bits == 4
    W = unpacked_qweight.clone().cpu()

    awq_scales = awq_scales.t().contiguous()
    unpacked_qzeros = unpacked_qzeros.contiguous()
    unpacked_qzeros = unpacked_qzeros.cpu()

    awq_scales = awq_scales.cpu()
    scale_zeros = unpacked_qzeros.t() * awq_scales
    scales = awq_scales.clone()

    infeatures = unpacked_qweight.shape[1]

    intweight = []
    for idx in range(infeatures):
        g_idx = idx // group_size

        intweight.append(
            torch.round((W[:, idx] + scale_zeros[:, g_idx]) / scales[:, g_idx]).to(
                torch.int
            )[:, None]
        )
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()
    intweight = intweight.numpy().astype(np.uint32)

    i = 0
    row = 0
    qweight = np.zeros(
        (intweight.shape[0] // 32 * bits, intweight.shape[1]), dtype=np.uint32
    )
    while row < qweight.shape[0]:
        for j in range(i, i + (32 // bits)):
            qweight[row] |= intweight[j] << (bits * (j - i))
        i += 32 // bits
        row += 1

    qweight = qweight.astype(np.int32)
    qweight = torch.from_numpy(qweight)

    unpacked_qzeros = unpacked_qzeros - 1
    torch.bitwise_and(unpacked_qzeros, (2**bits) - 1, out=unpacked_qzeros)

    unpacked_qzeros = unpacked_qzeros.numpy().astype(np.uint32)
    qzeros = np.zeros(
        (unpacked_qzeros.shape[0], unpacked_qzeros.shape[1] // 32 * bits),
        dtype=np.uint32,
    )
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        for j in range(i, i + (32 // bits)):
            qzeros[:, col] |= unpacked_qzeros[:, j] << (bits * (j - i))
        i += 32 // bits
        col += 1

    qzeros = qzeros.astype(np.int32)
    qzeros = torch.from_numpy(qzeros)

    return qweight, qzeros
