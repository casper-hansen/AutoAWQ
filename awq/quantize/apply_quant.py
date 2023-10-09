import torch

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, get_scale=False):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)

    if get_scale:
        return w, scales
    else:
        return w

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, get_scale=False):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)

    if get_scale:
        return w, scales
    else:
        return w

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, get_scale=False):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)

    if get_scale:
        return t, scales
    else:
        return t

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, get_scale=False):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)

    if get_scale:
        return t, scales
    else:
        return t
