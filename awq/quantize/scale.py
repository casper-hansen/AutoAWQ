import torch
import torch.nn as nn
from typing import Tuple, List
from awq.utils.utils import get_best_device
from awq.modules.act import ScaledActivation
from awq.utils.module import get_op_by_name, set_op_by_name
from transformers.models.bloom.modeling_bloom import BloomGelu
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.cohere.modeling_cohere import CohereLayerNorm
from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation

allowed_norms = [nn.LayerNorm, LlamaRMSNorm, GemmaRMSNorm, Gemma2RMSNorm, CohereLayerNorm]
allowed_act_fns = [
    nn.GELU,
    BloomGelu,
    NewGELUActivation,
    PytorchGELUTanh,
    GELUActivation,
]


@torch.no_grad()
def apply_clip(module, clip_list: Tuple[str, torch.Tensor]):
    for name, max_val in clip_list:
        layer: nn.Linear = get_op_by_name(module, name)
        layer.to(get_best_device())
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()


def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        best_device = get_best_device()
        prev_op.to(best_device)
        for layer in layers:
            layer.to(best_device)
        scales.to(best_device)

        if (
            isinstance(prev_op, nn.Linear)
            and type(layers) == list
            and isinstance(layers[0], nn.Linear)
        ):
            scale_fc_fcs(prev_op, layers, scales)

        elif isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)

        elif (
            any(isinstance(prev_op, t) for t in allowed_norms)
            or "rmsnorm" in str(prev_op.__class__).lower()
        ):
            scale_ln_fcs(prev_op, layers, scales)

        elif any(isinstance(prev_op, t) for t in allowed_act_fns):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)

        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                # Skip the modules that are not quantized
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()


@torch.no_grad()
def scale_ln_fcs(ln: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # GemmaRMSNorm is different from Llama's in that it multiplies
    # (1 + weight) to the output, instead of just weight.
    if isinstance(ln, GemmaRMSNorm) or isinstance(ln, Gemma2RMSNorm):
        ln.weight += 1
        ln.weight.div_(scales)
        ln.weight -= 1
    else:
        ln.weight.div_(scales)

    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fcs(fc1: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu: allowed_act_fns, fc: nn.Linear, scales: torch.Tensor):
    assert any(isinstance(gelu, t) for t in allowed_act_fns)
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0
