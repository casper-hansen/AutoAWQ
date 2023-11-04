from .base import BaseAWQForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as OldLlamaDecoderLayer,
    LlamaForCausalLM as OldLlamaForCausalLM
)

class LlamaAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldLlamaForCausalLM):
        fuser = NewLlamaFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldLlamaForCausalLM):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module: OldLlamaDecoderLayer):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model: OldLlamaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: OldLlamaDecoderLayer, input_feat, module_kwargs):
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
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaMLP

from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from awq.utils.fused_utils import fuse_qkv

class NewLlamaFuser:
    def __init__(self, model: OldLlamaForCausalLM):
        self.model = model

        self.llama_blocks = List[Tuple[str, OldLlamaDecoderLayer]] = [
            (name, module) for name, module in self.model.named_modules()
            if 'llamadecoderlayer' in module.__class__.__name__.lower()
        ]
    
    def fuse_transformer(self):
        blocks = []

        module: OldLlamaDecoderLayer
        for module in self.model.model.layers:
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj
            )
            blocks.append(LlamaLikeBlock(
                hidden_size=self.model.config.hidden_size,
                n_heads=self.model.config.num_attention_heads,
                n_kv_heads=self.model.config.num_key_value_heads,
                qkv_layer=qkv,
                o_proj=module.self_attn.o_proj,
                mlp=module.mlp,
                norm_1=module.input_layernorm,
                norm_2=module.post_attention_layernorm,
                dev=device,
                max_seq_len=self.model.config.max_new_tokens
            ))
        
        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )

class LlamaFuser:
    def __init__(self, model):
        self.model = model

        self.attention_modules: List[Tuple[str, LlamaAttention]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaAttention)
        ]

        self.rmsnorm_modules: List[Tuple[str, LlamaRMSNorm]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaRMSNorm)
        ]
        
        self.mlp_modules: List[Tuple[str, LlamaMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaMLP)
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

    def fuse_rmsnorm(self):
        for name, module in self.rmsnorm_modules:
            norm = FasterTransformerRMSNorm(module.weight, module.variance_epsilon)
            set_module_name(self.model, name, norm)

    def fuse_mlp(self):
        for name, module in self.mlp_modules:
            mlp = QuantLlamaMLP(module.gate_proj, module.down_proj, module.up_proj)
            set_module_name(self.model, name, mlp)