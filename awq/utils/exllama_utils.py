import torch


# non tensor for exllama kernels
none_tensor = torch.empty((1, 1), device="meta")


def awq_reverse_reorder_int_tensor(int_tensor, bits: int):
    compress_ratio = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert int_tensor.shape[-1] % compress_ratio == 0

    order_tensor = torch.tensor(
        order_map,
        dtype=torch.int32,
        device=int_tensor.device,
    )[None, :]

    order_tensor = order_tensor.repeat(int_tensor.shape[1] // compress_ratio, 1)

    order_tensor += torch.arange(
        0,
        int_tensor.shape[-1],
        compress_ratio,
        dtype=torch.int32,
        device=int_tensor.device,
    )[:, None]

    order_tensor = order_tensor.view(-1)

    reverse_order_tensor = torch.arange(
        order_tensor.shape[0],
        dtype=torch.int32,
        device=int_tensor.device,
    )[order_tensor][order_tensor]

    int_tensor = int_tensor[:, reverse_order_tensor]

    return int_tensor


def unpack(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    qweight = qweight.T
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :])
    izeros = izeros.reshape(-1, izeros.shape[1] * izeros.shape[2])

    iweights = torch.bitwise_right_shift(qweight[:, None, :], shifts[None, :, None])
    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    iweights = iweights.reshape(-1, iweights.shape[-1])

    return iweights, izeros


def awq_reverse_reorder(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    iweights = iweights.T

    izeros = awq_reverse_reorder_int_tensor(izeros, bits)
    iweights = awq_reverse_reorder_int_tensor(iweights, bits)

    return iweights, izeros


def pack(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    qweight = torch.zeros(
        (iweights.shape[0] // 32 * bits, iweights.shape[1]),
        dtype=torch.int32,
        device=iweights.device,
    )
    rows = torch.arange(
        qweight.shape[0],
        dtype=torch.int32,
        device=iweights.device,
    ) * (32 // bits)

    for j in range(32 // bits):
        qweight = torch.bitwise_or(
            qweight, torch.bitwise_left_shift(iweights[j + rows], (bits * j))
        )

    qzeros = torch.zeros(
        (izeros.shape[0], izeros.shape[1] // 32 * bits),
        dtype=torch.int32,
        device=izeros.device,
    )
    cols = torch.arange(
        qzeros.shape[1],
        dtype=torch.int32,
        device=izeros.device,
    ) * (32 // bits)

    for j in range(32 // bits):
        qzeros = torch.bitwise_or(
            qzeros,
            torch.bitwise_left_shift(izeros[:, j + cols], (bits * j)),
        )

    return qweight, qzeros


def exllama_post_init(model):
    for _, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
            submodule.post_init()

    return model


def exllamav2_post_init(model):
    fixed_bytes = {}
    for _, submodule in model.named_modules():
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
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
