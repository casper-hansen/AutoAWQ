import gc
from tqdm import auto as tqdm_lib

import torch
from torch import nn

from transformers.models.llama4.modeling_llama4 import (
    Llama4ForConditionalGeneration as OldLlama4ForConditionalGeneration, 
    Llama4TextDecoderLayer as OldLlama4TextDecoderLayer,
    Llama4TextMLP as OldLlama4TextMLP,
    Llama4TextMoe as OldLlama4TextMoe,
    Llama4TextExperts as OldLlama4TextExperts,
    Llama4TextMLP,
)
from transformers import AutoProcessor, AutoConfig, PreTrainedModel

from .base import BaseAWQForCausalLM
from awq.modules.act import ScaledActivation
from awq.utils.module import set_op_by_name

class Llama4TextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.dummy_fn = nn.Identity()
        self.up_proj = nn.Linear(self.hidden_size*self.num_experts, self.expert_dim, dtype=torch.float16)
        self.gate_proj = nn.Linear(self.hidden_size*self.num_experts, self.expert_dim, dtype=torch.float16)
        self.down_proj = nn.Linear(self.expert_dim, self.hidden_size*self.num_experts, dtype=torch.float16)
        self.act_fn = None
        self._quantization = False
        self._calib_data_seq_len = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        hidden_states = hidden_states.permute(1, 2, 0).reshape(-1, 1, self.hidden_size*self.num_experts)
        hidden_states = self.dummy_fn(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size*self.num_experts)
        down_proj = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        next_states = self.down_proj(down_proj)
        next_states = next_states.view(-1, self.hidden_size, self.num_experts)
        next_states = next_states.permute(2, 0, 1)
        return next_states.reshape(-1, self.hidden_size)

    @classmethod
    def replace(cls, llama4_text_experts: OldLlama4TextExperts):
        class Config:
            def __init__(self, num_local_experts, intermediate_size, hidden_size, hidden_act):
                self.num_local_experts = num_local_experts
                self.intermediate_size = intermediate_size
                self.hidden_size = hidden_size
                self.hidden_act = hidden_act
        
        config = Config(
            llama4_text_experts.num_experts,
            llama4_text_experts.intermediate_size,
            llama4_text_experts.hidden_size,
            'silu'
        )
        
        experts = cls(config)
        
        gate_up_proj = llama4_text_experts.gate_up_proj
        down_proj = llama4_text_experts.down_proj
        is_meta = hasattr(gate_up_proj, 'device') and gate_up_proj.device.type == 'meta'
        experts.act_fn = llama4_text_experts.act_fn
        if not is_meta:
            gate_part = gate_up_proj[:, :, :llama4_text_experts.expert_dim]
            up_part = gate_up_proj[:, :, llama4_text_experts.expert_dim:]
            
            gate_weight = gate_part.permute(2, 1, 0)
            gate_weight = gate_weight.reshape(llama4_text_experts.expert_dim, llama4_text_experts.hidden_size * llama4_text_experts.num_experts)
            up_weight = up_part.permute(2, 1, 0)
            up_weight = up_weight.reshape(llama4_text_experts.expert_dim, llama4_text_experts.hidden_size * llama4_text_experts.num_experts)
            down_weight = down_proj.permute(1, 2, 0)
            down_weight = down_weight.reshape(llama4_text_experts.hidden_size * llama4_text_experts.num_experts, llama4_text_experts.expert_dim)
            
            experts.gate_proj.weight.data.copy_(gate_weight)
            experts.up_proj.weight.data.copy_(up_weight)
            experts.down_proj.weight.data.copy_(down_weight)
            llama4_text_experts = llama4_text_experts.to('cpu', torch.float16)
        
        del llama4_text_experts
        gc.collect()
        
        return experts

class Llama4TextMoe(OldLlama4TextMoe):
    def __init__(self, config):
        self._quantization_mode = False
        self._seq_len = None
        super().__init__(config)

    def forward(self, hidden_states):
        # Due to the implementation of llama4, when trying to process with AutoAWQ, an error occurs on the second forward call.
        # Implementing a workaround for this issue.
        if not self._quantization_mode:
            self._seq_len = hidden_states.shape[1]
            self._quantization_mode = True
        elif len(hidden_states.shape)==2:
            hidden_states = hidden_states.view(-1, self._seq_len, self.hidden_dim)
        else:
            hidden_states = hidden_states.view(-1, self._seq_len, self.hidden_dim)
        out, router_scores = super().forward(hidden_states)
        #return out.view(self.num_experts, -1, self.hidden_dim), router_scores.view(self.num_experts, -1, self.hidden_dim)
        return out, router_scores

    @classmethod
    def replace(cls, llama4_text_moe: OldLlama4TextMoe):
        class Config:
            def __init__(self, hidden_size, num_local_experts, num_experts_per_tok):
                self.hidden_size = hidden_size
                self.num_local_experts = num_local_experts
                self.num_experts_per_tok = num_experts_per_tok
                self.intermediate_size = llama4_text_moe.shared_expert.up_proj.out_features
                self.hidden_act = "silu"
        
        config = Config(
            hidden_size=llama4_text_moe.hidden_dim,
            num_local_experts=llama4_text_moe.num_experts,
            num_experts_per_tok=llama4_text_moe.top_k
        )
        
        moe = cls(config)
        
        moe.router = llama4_text_moe.router
        moe.shared_expert = llama4_text_moe.shared_expert
        
        old_experts = llama4_text_moe.experts
        is_meta = False
        if hasattr(llama4_text_moe, 'hidden_dim'):
            tensor = next(llama4_text_moe.parameters(), None)
            if tensor is not None:
                is_meta = tensor.device.type == 'meta'
        
        # Don't try to move meta tensors to CPU
        if not is_meta:
            llama4_text_moe = llama4_text_moe.to('cpu')
        del llama4_text_moe
        gc.collect()
        
        moe.experts = Llama4TextExperts.replace(old_experts)
        
        return moe.to(torch.float16)

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
        scales = []
        if isinstance(module.feed_forward, OldLlama4TextMoe):
            scales.append(
                dict(
                    scale_name="feed_forward.shared_expert.activation_fn",
                    scale_layer=module.feed_forward.shared_expert.activation_fn,
                    scale_shape=module.feed_forward.shared_expert.gate_proj.out_features
                )
            )
            scales.append(
                dict(
                    scale_name=f"feed_forward.experts.act_fn",
                    scale_layer=module.feed_forward.experts.act_fn,
                    scale_shape=module.feed_forward.experts.gate_proj.out_features
                )
            )
            scales.append(
                dict(
                    scale_name=f"feed_forward.experts.dummy_fn",
                    scale_layer=module.feed_forward.experts.dummy_fn,
                    scale_shape=module.feed_forward.experts.gate_proj.in_features
                )
            )
            
        elif isinstance(module.feed_forward, OldLlama4TextMLP):
            scales.append(
                dict(
                    scale_name="feed_forward.activation_fn",
                    scale_layer=module.feed_forward.activation_fn,
                    scale_shape=module.feed_forward.gate_proj.out_features
                )
            )
        
        return dict(is_scalable=True, scales = scales)
                
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
        
        # Attention Block
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
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        if isinstance(module.feed_forward, Llama4TextMoe):
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.feed_forward.shared_expert.gate_proj, 
                            module.feed_forward.shared_expert.up_proj],
                    inp=input_feat["feed_forward.shared_expert.gate_proj"],
                    module2inspect=module.feed_forward,
                )
            )
            
            layers.append(
                dict(
                    prev_op=module.feed_forward.experts.dummy_fn,
                    layers=[module.feed_forward.experts.gate_proj, 
                            module.feed_forward.experts.up_proj],
                    inp=input_feat["feed_forward.experts.gate_proj"],
                    module2inspect=module.feed_forward.experts,
                )
            )
            layers.append(
                dict(
                    prev_op=module.feed_forward.shared_expert.activation_fn,
                    layers=[module.feed_forward.shared_expert.down_proj],
                    inp=input_feat["feed_forward.shared_expert.down_proj"],
                )
            )
            layers.append(
                dict(
                    prev_op=module.feed_forward.experts.act_fn,
                    layers=[module.feed_forward.experts.down_proj],
                    inp=input_feat["feed_forward.experts.down_proj"],
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
        assert not (
            version == "gemv" and (use_exllama or use_exllama_v2 or use_ipex)
        ), "Exllama kernels only support GEMM version."
        
        # Get blocks of model
        layers = self.get_model_layers(model)

        for layer in tqdm_lib.tqdm(layers, desc="Replacing MoE Block..."):
            if isinstance(layer.feed_forward, OldLlama4TextMoe):
                layer.feed_forward = Llama4TextMoe.replace(layer.feed_forward)
            gc.collect()
        model.tie_weights()
        super()._load_quantized_modules(
            self, model=model, quant_config=quant_config, version=version, use_exllama=use_exllama, use_exllama_v2=use_exllama_v2, use_ipex=use_ipex
        )

    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)
        scales = scale_dict.get('scales')
        if scale_dict["is_scalable"] and scales is None:
            scales = [scale_dict]
        if scale_dict["is_scalable"]:
            for scale in scales:
                if not isinstance(scale["scale_layer"], ScaledActivation):
                    param = next(layer.parameters())
    
                    # get activation scale
                    scale_like = torch.ones(
                        scale["scale_shape"], dtype=param.dtype, device=param.device
                    )
    
                    # scale activation
                    scaled_act = ScaledActivation(scale["scale_layer"], scale_like)
                    set_op_by_name(layer, scale["scale_name"], scaled_act)
