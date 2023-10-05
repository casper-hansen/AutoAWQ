from typing import Dict
from .base import BaseAWQForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralForCausalLM

class MistralAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MistralDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: MistralForCausalLM, quant_config: Dict):
        fuser = MistralFuser(model, quant_config)
        fuser.fuse_attention()
        fuser.fuse_rmsnorm()
        fuser.fuse_mlp()
    
    @staticmethod
    def get_model_layers(model: MistralForCausalLM):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module: MistralDecoderLayer):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model: MistralForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: MistralDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))

        return layers

import torch
from typing import List, Tuple, Union
from awq.utils.utils import set_module_name
from awq.modules.fused.mlp import QuantLlamaMLP
from awq.modules.fused.attn import QuantAttentionFused
from awq.modules.fused.norm import FasterTransformerRMSNorm
from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRMSNorm, MistralMLP

class MistralFuser:
    def __init__(self, model, quant_config):
        self.model = model
        self.quant_config = quant_config

        self.attention_modules: List[Tuple[str, MistralAttention]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, MistralAttention)
        ]

        self.rmsnorm_modules: List[Tuple[str, MistralRMSNorm]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, MistralRMSNorm)
        ]
        
        self.mlp_modules: List[Tuple[str, MistralMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, MistralMLP)
        ]
    
    def fuse_attention(self):
        for name, module in self.attention_modules:
            qkv_layer: Union[WQLinear_GEMM, WQLinear_GEMV] = self._fuse_qkv(module)
            attn = QuantAttentionFused(
                module.hidden_size,
                module.num_heads,
                module.num_key_value_heads,
                qkv_layer, 
                module.o_proj,
                next(iter(qkv_layer.state_dict().values())).device,
                self.model.config.max_new_tokens
            )
            set_module_name(self.model, name, attn)
    
    def _fuse_qkv(self, module: MistralAttention):
        q_proj, k_proj, v_proj = module.q_proj, module.k_proj, module.v_proj
        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

        if isinstance(q_proj, WQLinear_GEMV):
            q_linear = WQLinear_GEMV
        else:
            q_linear = WQLinear_GEMM

        qkv_layer = q_linear(
            q_proj.w_bit,
            q_proj.group_size,
            q_proj.in_features,
            q_proj.out_features + k_proj.out_features + v_proj.out_features,
            q_proj.bias is not None,
            next(iter(module.state_dict().values())).device
        )

        if isinstance(qkv_layer, WQLinear_GEMV):
            qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=0)
            qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=0)
            qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=0)
            qkv_layer.split_k_iters = q_proj.split_k_iters
        else:
            qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
            qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
            qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)
        
        qkv_layer.bias = bias

        return qkv_layer

    def fuse_rmsnorm(self):
        for name, module in self.rmsnorm_modules:
            norm = FasterTransformerRMSNorm(module.weight, module.variance_epsilon)
            set_module_name(self.model, name, norm)

    def fuse_mlp(self):
        for name, module in self.mlp_modules:
            mlp = QuantLlamaMLP(module.gate_proj, module.down_proj, module.up_proj)
            set_module_name(self.model, name, mlp)
