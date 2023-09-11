from .base import BaseAWQForCausalLM
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconForCausalLM, FalconAttention

class FalconAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "FalconDecoderLayer"

    @staticmethod
    def fuse_layers(model: FalconForCausalLM, quant_config:dict):
        fuser = FalconFuser(model)

    @staticmethod
    def get_model_layers(model: FalconForCausalLM):
        return model.transformer.h
    
    @staticmethod
    def get_act_for_scaling(module: FalconDecoderLayer):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.dense_h_to_4h.out_features
        )
    
    @staticmethod
    def move_embed(model: FalconForCausalLM, device):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: FalconDecoderLayer, input_feat, module_kwargs):
        layers = []
        
        # Falcon 7B (older architecture)
        if module.config.num_attention_heads == 71:
            # linear 1 + attention
            layers.append(dict(
                prev_op=module.input_layernorm,
                layers=[module.mlp.dense_h_to_4h, module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

        # Falcon 40B (newer architecture)
        else:
            # linear 1 + attention
            layers.append(dict(
                prev_op=module.ln_attn,
                layers=[module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

            # linear 2
            layers.append(dict(
                prev_op=module.ln_mlp,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat['mlp.dense_h_to_4h'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

        return layers

import torch
from torch.nn import LayerNorm
from typing import List, Tuple
from awq.utils.utils import set_module_name
from awq.modules.fused.attn import QuantAttentionFused

class FalconFuser:
    def __init__(self, model):
        self.model = model

        self.attention_modules: List[Tuple[str, FalconAttention]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, FalconAttention)
        ]

        self.layernorm_modules: List[Tuple[str, LayerNorm]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LayerNorm)
        ]
    
    def fuse_attention(self):
        for name, qkv_layer in self.attention_modules:
            attn = QuantAttentionFused(
                qkv_layer.hidden_size,
                qkv_layer.num_heads,
                qkv_layer, 
                qkv_layer.dense,
                next(iter(qkv_layer.state_dict().values())).device,
                self.model.config.max_new_tokens
            )
            set_module_name(self.model, name, attn)