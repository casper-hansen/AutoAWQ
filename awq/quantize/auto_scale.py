import gc
import torch
import torch.nn as nn
import logging

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import NewGELUActivation
from awq.modules.act import ScaledActivation
from awq.utils.module import get_op_by_name, get_op_name, set_op_by_name

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]
    
    scales = scales.to(ln.weight.device)

    # debugging start even scales = 1 does not work?
    """
    scales = scales * 0
    scales = scales + 1
    """
    # debugging end

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features
    
    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert any(isinstance(gelu,t) for t in [nn.GELU, BloomGelu, NewGELUActivation])
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0

def pseudo_quantize_tensor(w, w_bit=4,
                           zero_point=True, 
                           q_group_size=-1,
                           inplace=False,
                           get_scale_zp=False
                           ):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** w_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (w_bit - 1) - 1
        min_int = - 2 ** (w_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w    

@torch.no_grad()
def auto_scale_block(awq_model,
                     module, 
                     module_kwargs,
                     quant_config,
                     input_feat):
    # from .quantizer import pseudo_quantize_tensor
    # firstly, get the weight quantize function
    if quant_config['w_bit'] is not None:
        def w_quantize_func(p): return pseudo_quantize_tensor(p, w_bit=quant_config["w_bit"], q_group_size=quant_config["q_group_size"]).detach()
    else:
        def w_quantize_func(p): return p

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # find the best scale ratio
    def _search_module_scale(module2inspect, layers: list, inp, kwargs={}):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]
            
        # w: co, ci
        # x: n, ci
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        weight = weight.view(-1, quant_config.get("q_group_size"))
        w_scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
        w_scale = w_scale.view(org_shape)
        w_max = w_scale.mean(0)

        # Clear GPU memory
        del weight
        gc.collect()
        torch.cuda.empty_cache()

        inp = inp.to(next(module2inspect.parameters()).device)
        with torch.no_grad():
            org_out = module2inspect(inp, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(inp)

        best_error = float('inf')
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = (x_max.pow(ratio) / w_max.pow(1-ratio)
                      ).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in layers:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(
                    fc.weight.data) / (scales.view(1, -1))
            out = module2inspect(inp, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            module2inspect.load_state_dict(org_sd)
        if best_ratio == -1:
            logging.debug(history)
            raise Exception
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):
        scales = _search_module_scale(module2inspect, layers, inp, kwargs)
        scales = scales.detach().cpu()
        
        # prev_op_name, [layer_name], scale
        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), scales)

    layers: list[dict] = awq_model.get_layers_for_scaling(
        module, input_feat, module_kwargs
    )
    scales_list = [_auto_get_scale(**layer) for layer in layers]

    return scales_list

def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()
        
        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)

        elif any(isinstance(prev_op,t) for t in [nn.LayerNorm, LlamaRMSNorm]) \
             or 'rmsnorm' in str(prev_op.__class__).lower():
            scale_ln_fcs(prev_op, layers, scales)

        elif any(isinstance(prev_op,t) for t in [nn.GELU, BloomGelu, NewGELUActivation]):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
            
        else:
            raise NotImplementedError(
                f"prev_op {type(prev_op)} not supported yet!")
            
        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:  
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()
