import torch
import inspect
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from awq.utils.utils import clear_memory
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip
from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize
)


class AwqQuantizer:
    def __init__(self, awq_model, model, tokenizer, w_bit, group_size, version, 
                       calib_data, split, text_column, duo_scaling, modules_to_not_convert=None) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.modules_to_not_convert = modules_to_not_convert if modules_to_not_convert is not None else []
        self.modules, self.module_kwargs, self.inps = self.init_quant()
    
    def pseudo_quantize_tensor(self, w: torch.Tensor, get_scale_zp=False):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2

        # zero point quantization
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** self.w_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
        assert torch.isnan(w).sum() == 0

        w = w.reshape(org_w_shape)

        if get_scale_zp:
            return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
        else:
            return w
    
    def pseudo_dequantize_tensor(self, w: nn.Linear, scales: torch.Tensor, zeros: torch.Tensor):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // zeros.shape[-1]

        # get zeros and scales in correct shape
        zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        w = (w.weight.data - zeros) * scales

        return w
    
    def quantize(self):
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                self.modules[i] = self.modules[i].cuda()
                common_device = next(self.modules[i].parameters()).device
            
            self.inps = self.inps.to(common_device)

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(named_linears, self.modules_to_not_convert)

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [self._search_best_scale(self.modules[i], **layer) for layer in module_config]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(scales_list, get_op_name(self.model, self.modules[i]) + ".")

            # [STEP 3]: Compute and apply clipping list
            clip_list = self._search_best_clip(self.modules[i], named_linears, input_feat)
            apply_clip(self.modules[i], clip_list)
            clip_list = append_str_prefix(clip_list, get_op_name(self.model, self.modules[i]) + ".")

            # [STEP 4]: Quantize weights
            self._apply_quant(self.modules[i], named_linears)
            clear_memory()
    
    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.cuda().half()

            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data, 
                get_scale_zp=True
            )

            if self.version == 'GEMM':
                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM

            elif self.version  == 'GEMV':
                q_linear_module = WQLinear_GEMV
            
            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()

    @torch.no_grad()
    def _search_best_scale(self, module, prev_op, layers: List[nn.Linear], inp: torch.Tensor, module2inspect=None, kwargs={}):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]
        
        if "use_cache" in kwargs:
            kwargs.pop("use_cache")
        
        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute maximum of weight
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        weight = weight.view(-1, self.group_size)
        w_scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
        w_scale = w_scale.view(org_shape)
        w_max = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute maximum of x
        x_max = inp.abs().view(-1, inp.shape[-1]).mean(0)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)

            fp16_output = module2inspect(inp, **module_kwargs)
            if isinstance(fp16_output, tuple):
                fp16_output = fp16_output[0]
        
        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_max, x_max, module2inspect,
            layers, fp16_output, module_kwargs
        )
        
        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), best_scales)

    def _compute_best_scale(self, x, w_max, x_max, module2inspect, linears2scale: List[nn.Linear],
                                  fp16_output, kwargs={}):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float('inf')

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}
        
        device = x.device
        x_max = x_max.view(-1).to(device)
        w_max = w_max.view(-1).to(device)
        
        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_max.pow(ratio) / w_max.pow(1-ratio)).clamp(min=1e-4)
            else:
                scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = self.pseudo_quantize_tensor(fc.weight.data) / scales_view

            # W * X
            int_w_output = module2inspect(x, **kwargs)
            if isinstance(int_w_output, tuple):
                int_w_output = int_w_output[0]
            
            # compute mean squared error (L2 norm)
            loss = (fp16_output - int_w_output).float().pow(2).mean().item() # NOTE: float prevents overflow

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].cuda()
            max_val = self._compute_best_clip(named_linears[name].weight, input_feat[name])
            clip_list.append((name, max_val))

            named_linears[name].cpu()
        
        return clip_list

    @torch.no_grad()
    def _compute_best_clip(self, w: torch.Tensor, input_feat: torch.Tensor, n_grid=20, max_shrink=0.5, n_sample_token=512):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else w.shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
        w = w.reshape(w.shape[0], 1, -1, group_size)

        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = - max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def init_quant(self, n_samples=128, seqlen=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data, tokenizer=self.tokenizer, n_samples=n_samples, block_size=seqlen,
            split=self.split, text_column=self.text_column
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        modules[0] = modules[0].cuda()
        self.awq_model.move_embed(self.model, "cuda")
        
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        
        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        modules[0] = modules[0].module  # restore
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")
        
        clear_memory()
        
        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to("cuda")

        return modules, layer_kwargs, inps
    
    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {**named_linears, "block_sparse_moe": layer.block_sparse_moe}

        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                feat_dict=input_feat)))
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        
        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = layer(self.inps, **module_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        
        return input_feat


    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers. 

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in  inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs