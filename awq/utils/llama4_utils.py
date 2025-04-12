import functools
from collections import defaultdict
from tqdm import tqdm

from ..quantize.quantizer import AwqQuantizer
from ..quantize.scale import apply_clip
from utils import clear_memory, get_best_device
from calib_data import get_calib_dataset
from ..modules.act import ScaledActivation
from transformers.activations import ACT2FN

from transformers.models.llama4.modeling_llama4 import Llama4TextRMSNorm
from transformers.feature_extraction_utils import BatchFeature

from module import (
    append_str_prefix,
    get_op_name,
    get_op_by_name, 
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)
from modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_IPEX,
    WQLinear_Marlin,
    WQLinear_Exllama,
    WQLinear_ExllamaV2,
    WQLinear_GEMVFast,
    marlin_post_init,
    exllama_post_init,
    exllamav2_post_init,
    ipex_post_init,
)

class Llama4AwqQuantizer(AwqQuantizer):
        
    @torch.no_grad()
    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

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
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        return modules, layer_kwargs, inps

    def _preprocess(self):
        pass

    def _preprocess_layer_iter(self, layer_index):
        if isinstance(self.modules[layer_index].feed_forward, OldLlama4TextMoe):
            self.modules[layer_index].feed_forward = Llama4TextMoe.replace(self.modules[layer_index].feed_forward.to('cpu'))
        
        common_device = next(self.modules[layer_index].parameters()).device
        if common_device is None or str(common_device) == "cpu":
            if torch.cuda.is_available():
                best_device = "cuda:" + str(layer_index % torch.cuda.device_count())
            else:
                best_device = get_best_device()

            self.modules[layer_index] = self.modules[layer_index].to(best_device)
            common_device = next(self.modules[layer_index].parameters()).device

        for k in self.inps_dict.keys():
            if self.module_kwargs_dict[k].get("position_ids") is not None:
                self.module_kwargs_dict[k]["position_ids"] = self.module_kwargs_dict[k][
                    "position_ids"
                ].to(common_device)
    
            if self.module_kwargs_dict[k].get("attention_mask") is not None:
                self.module_kwargs_dict[k]["attention_mask"] = self.module_kwargs_dict[k][
                    "attention_mask"
                ].to(common_device)
    
            self.inps_dict[k] = self.inps_dict[k].to("cpu")

    def _postprocess_layer_iter(self, layer_index):
        self.modules[layer_index] = self.modules[layer_index].to("cpu")
        torch.save(self.modules[layer_index].state_dict(), f"/workspace/mllama-autoawq/tmp/q{layer_index:04}.pt")
        self.modules[layer_index] = None
    
    def quantize(self):
        self._preprocess()
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            self._preprocess_layer_iter(i)
            
            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)
            self._postprocess_layer_iter(i)

            clear_memory()
    