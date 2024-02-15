import torch
import torch.nn as nn
import numpy as np

try:
    import marlin_cuda  # with CUDA kernels (AutoAWQ_kernels)

    MARLIN_INSTALLED = True
except:
    MARLIN_INSTALLED = False


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)

        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()


class WQLinear_Marlin(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.w_bit = w_bit
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size != -1 else in_features
        self.max_par = 8  # partitioning for large inputs

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        ######################################################
        ## These shapes are only specific for Marlin models ##
        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features // 16, out_features * 16 // 8),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        ######################################################

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

    @classmethod
    def from_linear(
        cls,
        linear,
        w_bit,
        group_size,
        init_only=False,
        scales=None,
        zeros=None,
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

        assert zeros is None and scales is not None

        tile = 16
        maxq = 2**4 - 1
        s = scales.t()
        w = linear.weight.data.t()
        if awq_linear.group_size != awq_linear.in_features:
            w = w.reshape((-1, awq_linear.group_size, awq_linear.out_features))
            w = w.permute(1, 0, 2)
            w = w.reshape((awq_linear.group_size, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if awq_linear.group_size != awq_linear.in_features:
            w = w.reshape((awq_linear.group_size, -1, awq_linear.out_features))
            w = w.permute(1, 0, 2)
            w = w.reshape(
                (awq_linear.in_features, awq_linear.out_features)
            ).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, awq_linear.out_features)).contiguous()
        w = w.reshape(
            (
                awq_linear.in_features // tile,
                tile,
                awq_linear.out_features // tile,
                tile,
            )
        )
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((awq_linear.in_features // tile, awq_linear.out_features * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        awq_linear.qweight[:] = q.to(awq_linear.qweight.device)
        awq_linear.scales[:] = s.to(awq_linear.qweight.device)

        if awq_linear.bias is not None:
            awq_linear.bias[:] = linear.bias.data.to(awq_linear.bias.device)

        return awq_linear

    def post_init(self):
        self.register_buffer(
            "workspace",
            torch.zeros(
                self.out_features // 128 * self.max_par,
                dtype=torch.int32,
                device=self.qweight.device,
            ),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, x):
        assert hasattr(self, "workspace"), (
            "module.post_init() must be called before module.forward(). "
            "Use marlin_post_init() on the whole model."
        )
        assert MARLIN_INSTALLED, (
            "Marlin kernels are not installed. "
            "Please install AWQ compatible Marlin kernels from AutoAWQ_kernels."
        )

        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        x = x.view(-1, x.shape[-1])

        out = torch.empty(
            (x.shape[0], self.out_features),
            dtype=torch.float16,
            device=x.device,
        )
        marlin_cuda.mul(
            x,
            self.qweight,
            out,
            self.scales,
            self.workspace,
            -1,  # thread_k
            -1,  # thread_n
            -1,  # sms
            self.max_par,
        )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        if self.bias is not None:
            out.add_(self.bias)

        return out.view(out_shape)

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )


def marlin_post_init(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, WQLinear_Marlin):
            submodule.post_init()

    return model
