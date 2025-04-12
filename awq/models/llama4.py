from typing import Dict, List, Optional
import gc
from tqdm import auto as tqdm_lib

import torch
from torch import nn

from transformers.models.llama4.modeling_llama4 import (
    Llama4ForConditionalGeneration as OldLlama4ForConditionalGeneration, 
    Llama4TextDecoderLayer as OldLlama4TextDecoderLayer,
    Llama4TextMLP as OldLlama4TextMLP,
    Llama4TextMoe as OldLlama4TextMoe,
    Llama4TextMLP,
)
from transformers import AutoProcessor, AutoConfig

from accelerate.big_modeling import init_empty_weights

from .base import BaseAWQForCausalLM

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
    def replace(cls, original_moe_block: OldLlama4TextMoe):
        param = next(original_moe_block.parameters())
        device = param.device
        
        config = original_moe_block.shared_expert.config
        moe = cls(config)
        moe.router = original_moe_block.router
        moe.shared_expert = original_moe_block.shared_expert

        _gate_up_proj = original_moe_block.experts.state_dict()['gate_up_proj']
        down_param = original_moe_block.experts.state_dict()['down_proj']
        gate_param, up_param = _gate_up_proj.chunk(2, dim=-1)

        for i in range(moe.num_experts):
            moe.experts[i].gate_proj.weight.data = gate_param[i].T
            moe.experts[i].up_proj.weight.data = up_param[i].T
            moe.experts[i].down_proj.weight.data = down_param[i].T
        original_moe_block = original_moe_block.to('cpu')
        del original_moe_block
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

        layers = self.get_model_layers(model.model)

        for layer in tqdm_lib.tqdm(layers, desc="Replacing MoE Block..."):
            if isinstance(layer.feed_forward, OldLlama4TextMoe):
                layer.feed_forward = Llama4TextMoe.replace(layer.feed_forward)
            gc.collect()
            
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
        layers = self.get_model_layers(model.model)

        for layer in tqdm_lib.tqdm(layers, desc="Replacing MoE Block..."):
            if isinstance(layer.feed_forward, OldLlama4TextMoe):
                with init_empty_weights():
                    layer.feed_forward = Llama4TextMoe.from_config(model.config.text_config)
            gc.collect()
        super()._load_quantized_modules(
            model, quant_config, version, use_exllama, use_exllama_v2, use_ipex=False
        )
