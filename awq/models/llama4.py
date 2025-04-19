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


class Llama4TextMLP(OldLlama4TextMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config, intermediate_size)
        self.dummy_fn = nn.Identity()

    def forward(self, x):
        x = self.dummy_fn(x)
        return super().forward(x)

class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        self.experts = nn.ModuleList([Llama4TextMLP(config) for _ in range(self.num_experts)])
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False, dtype=torch.float16)
        self.shared_expert = Llama4TextMLP(config)

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
        # We do this to make sure we have -inf for non topK tokens before going through the !
        # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
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
        routed_in = routed_in.view(self.num_experts, -1, self.hidden_dim)
        for i in range(self.num_experts):
            out += self.experts[i](routed_in[i])
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
            hidden_size        = llama4_text_moe.hidden_dim,
            num_local_experts  = llama4_text_moe.num_experts,
            num_experts_per_tok= llama4_text_moe.top_k,
        )
        moe = cls(config)

        moe.router        = llama4_text_moe.router
        moe.shared_expert = llama4_text_moe.shared_expert

        # meta デバイスのときはコピー不可なので退避なし
        is_meta = any(p.device.type == "meta" for p in llama4_text_moe.parameters())
        if not is_meta:
            llama4_text_moe = llama4_text_moe.to("cpu")
            
        old_experts = llama4_text_moe.experts
        hidden, inter = config.hidden_size, config.intermediate_size

        with torch.no_grad():
            for exp_id in range(config.num_local_experts):
                mlp = Llama4TextMLP(config).to(torch.float16)
    
                old_gate_up = old_experts.gate_up_proj[exp_id]
                old_down    = old_experts.down_proj[exp_id]
                mlp.gate_proj.weight = nn.Parameter(old_gate_up[:, :inter].T.contiguous())
                mlp.up_proj.weight = nn.Parameter(old_gate_up[:, inter:].T.contiguous())
                mlp.down_proj.weight = nn.Parameter(old_down.T.contiguous())
                moe.experts[exp_id] = mlp

        del llama4_text_moe
        gc.collect()

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
        if isinstance(module.feed_forward, Llama4TextMoe):
            scales.append(
                dict(
                    scale_name="feed_forward.shared_expert.activation_fn",
                    scale_layer=module.feed_forward.shared_expert.activation_fn,
                    scale_shape=module.feed_forward.shared_expert.gate_proj.out_features
                )
            )
            for i in range(module.feed_forward.num_experts):
                scales.append(
                    dict(
                        scale_name=f"feed_forward.experts.{i}.activation_fn",
                        scale_layer=module.feed_forward.experts[i].activation_fn,
                        scale_shape=module.feed_forward.experts[i].gate_proj.out_features
                    )
                )
                scales.append(
                    dict(
                        scale_name=f"feed_forward.experts.{i}.dummy_fn",
                        scale_layer=module.feed_forward.experts[i].dummy_fn,
                        scale_shape=module.feed_forward.experts[i].gate_proj.in_features
                    )
                )
            
        elif isinstance(module.feed_forward, Llama4TextMLP):
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
                    module2inspect=module.feed_forward.shared_expert,
                )
            )
            layers.append(
                dict(
                    prev_op=module.feed_forward.shared_expert.activation_fn,
                    layers=[module.feed_forward.shared_expert.down_proj],
                    inp=input_feat["feed_forward.shared_expert.down_proj"],
                )
            )
            
            for i in range(module.feed_forward.num_experts):
                layers.append(
                    dict(
                        prev_op=module.feed_forward.experts[i].dummy_fn,
                        layers=[module.feed_forward.experts[i].gate_proj, 
                                module.feed_forward.experts[i].up_proj],
                        inp=input_feat[f"feed_forward.experts.{i}.gate_proj"],
                        module2inspect=module.feed_forward.experts[i],
                    )
                )
                layers.append(
                    dict(
                        prev_op=module.feed_forward.experts[i].activation_fn,
                        layers=[module.feed_forward.experts[i].down_proj],
                        inp=input_feat[f"feed_forward.experts.{i}.down_proj"],
                    )
                )
        elif isinstance(module.feed_forward, Llama4TextMLP):
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
