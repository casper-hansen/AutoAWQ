import tqdm
from typing import Dict, List, Optional
import functools
from collections import defaultdict

import torch
from torch import nn

from transformers.models.llama4.modeling_llama4 import (
    Llama4ForConditionalGeneration as OldLlama4ForConditionalGeneration, 
    Llama4TextDecoderLayer as OldLlama4TextDecoderLayer,
    Llama4TextMLP as OldLlama4TextMLP,
    Llama4TextMoe as OldLlama4TextMoe,
    Llama4TextRMSNorm
)
from transformers.models.llama4.configuration_llama4 import Llama4Config
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoProcessor, AutoConfig
from transformers.activations import ACT2FN

from accelerate.big_modeling import init_empty_weights

from .base import BaseAWQForCausalLM
from awq.quantize.quantizer import AwqQuantizer
from awq.quantize.scale import apply_clip
from awq.utils.utils import clear_memory, get_best_device
from awq.utils.calib_data import get_calib_dataset
from awq.modules.act import ScaledActivation

from ..utils.module import (
    append_str_prefix,
    get_op_name,
    get_op_by_name, 
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)
from ..modules.linear import (
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

class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        models = []
        for i in range(self.num_experts):
            models.append(Llama4TextMLP(config).to(config.torch_dtype))
        self.experts = nn.ModuleList(models)
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False, dtype=config.torch_dtype)
        self.shared_expert = Llama4TextMLP(config).to(config.torch_dtype)

    @classmethod
    def replace(cls, original_inst: OldLlama4TextMoe):
        param = next(l.parameters())
        device = param.device
        
        config = original_inst.shared_expert.config
        moe = cls(config)
        moe.router = original_inst.router
        moe.shared_expert = original_inst.shared_expert

        _gate_up_proj = original_inst.experts.state_dict()['gate_up_proj']
        down_param = original_inst.experts.state_dict()['down_proj']
        gate_param, up_param = _gate_up_proj.chunk(2, dim=-1)

        for i in range(moe.num_experts):
            moe.experts[i].gate_proj.weight.data = gate_param[i].T
            moe.experts[i].up_proj.weight.data = up_param[i].T
            moe.experts[i].down_proj.weight.data = down_param[i].T
        original_inst = original_inst.to('cpu')
        del original_inst
        return moe.to(device)

    def forward(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states).transpose(0, 1)
        tokens_per_expert = batch * seq_len

        router_top_value, router_indices = torch.topk(router_logits.transpose(0, 1), self.top_k, dim=1)
        router_scores = (
            torch.full_like(router_logits.transpose(0, 1), float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )

        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device).view(1, -1).expand(router_scores.size(0), -1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
        
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)
        routed_in = routed_in * router_scores.reshape(-1, 1)
        out = self.shared_expert(hidden_states)
        for i in range(self.num_experts):
            if (routed_in[i:i+1,:]==0.0).sum()==0:
                continue
            out += self.experts[i](routed_in[i:i+1,:])

        return out, router_scores

class Llama4AWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = "Llama4TextDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model):
        raise NotImplementedError()

    @staticmethod
    def get_model_layers(model: OldLlama4ForConditionalGeneration):
        return model.language_model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldLlama4TextDecoderLayer):
        raise NotImplementedError()
                
    @staticmethod
    def move_embed(model: OldLlama4ForConditionalGeneration, device: str):
        m_list = [
            model.vision_model,
            model.multi_modal_projector,
            model.language_model.model.rotary_emb,
            model.language_model.model.norm,
            model.language_model.model.embed_tokens,
            model.language_model.lm_head,
        ]
        for m in m_list:
            m = m.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldLlama4TextDecoderLayer, input_feat, module_kwargs):
        layers = []
    
        assert isinstance(module, OldLlama4TextDecoderLayer)
        if "self_attn.q_proj" in input_feat.keys():
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )
        
        if "self_attn.o_proj" in input_feat.keys() and module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        if isinstance(module.feed_forward, OldLlama4TextMoe):
            tmp_layers = [module.feed_forward.shared_expert.gate_proj,
                          module.feed_forward.shared_expert.up_proj,
                         ]
            for l in module.feed_forward.experts:
                tmp_layers.append(l.up_proj)
                tmp_layers.append(l.gate_proj)
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=tmp_layers,
                    inp=input_feat["feed_forward.shared_expert.gate_proj"],
                    module2inspect=module.feed_forward,
                )
            )
            layers.append(
                dict(
                    prev_op=module.feed_forward.shared_expert.activation_fn,
                    layers=[module.feed_forward.shared_expert.down_proj],
                    inp=input_feat["feed_forward.shared_expert.down_proj"],
                )
            )
            for i in len(module.feed_forward.experts):
                layers.append(
                    dict(
                        prev_op=module.feed_forward.experts[i].act_fn,
                        layers=[module.feed_forward.experts[i].down_proj],
                        inp=input_feat[f"feed_forward.experts.{i}.down_proj"],
                    )
                )
            
        elif isinstance(module.feed_forward, OldLlama4TextMLP):
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.feed_forward.gate_proj,
                            module.feed_forward.up_proj,
                           ],
                    inp=input_feat["feed_forward.gate_proj"],
                    module2inspect=module.feed_forward,
                )
            )
            layers.append(
                dict(
                    prev_op=module.feed_forward.activation_fn,
                    layers=[module.feed_forward.down_proj],
                    inp=input_feat["feed_forward.down_proj"],
                )
            )
    
        return layers

    @classmethod
    def from_pretrained(
        self,
        model_path,
        model_type,
        torch_dtype = torch.float16,
        trust_remote_code = True,
        safetensors = True,
        device_map = "auto",
        download_kwargs = None,
        low_cpu_mem_usage = True,
        use_cache = False,
        **model_init_kwargs,
    ):
        if "config" not in model_init_kwargs.keys():
            model_init_kwargs["config"] = AutoConfig.from_pretrained(model_path)
            model_init_kwargs["config"].text_config.use_cache = use_cache
        else:
            model_init_kwargs["config"].text_config.use_cache = use_cache
        model_init_kwargs["config"].torch_dtype = torch_dtype
        model_init_kwargs["config"].vision_config.torch_dtype = torch_dtype
        model_init_kwargs["config"].text_config.torch_dtype = torch_dtype
        
        model = super().from_pretrained(
            model_path,
            model_type,
            torch_dtype = torch_dtype,
            trust_remote_code = trust_remote_code,
            safetensors = safetensors,
            device_map = device_map,
            download_kwargs = download_kwargs,
            low_cpu_mem_usage = low_cpu_mem_usage,
            **model_init_kwargs,
        )
        model.processor = AutoProcessor.from_pretrained(model_path)
        return model

    def _load_quantized_modules(
        self, model, quant_config, version, use_exllama, use_exllama_v2, use_ipex=False
    ):
        # Real quantization of weights
        assert not (
            version == "gemv" and (use_exllama or use_exllama_v2 or use_ipex)
        ), "Exllama kernels only support GEMM version."
        
        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing MoE Block..."):
            layer = layers[i]
            if isinstance(layer.feed_forward, OldLlama4TextMoe):
                with init_empty_weights():
                    layer.feed_forward = Llama4TextMoe.from_config(model.config.text_config)
            gc.collect()
        super()._load_quantized_modules(
            model, quant_config, version, use_exllama, use_exllama_v2, use_ipex=False
        )

class Llama4AwqQuantizer(AwqQuantizer):
    def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        w_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        modules_to_not_convert=None,
        export_compatible=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.export_compatible = export_compatible
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs_dict, self.inps_dict = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )
        
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
    