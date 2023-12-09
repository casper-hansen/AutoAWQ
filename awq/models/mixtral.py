import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import MixtralBlock
from awq.modules.fused.model import LlamaLikeModel
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer as OldMistralDecoderLayer,
    MistralForCausalLM as OldMistralForCausalLM
)
from awq.modules.fused.mlp import QuantFusedMLP
from awq.modules.fused.norm import FasterTransformerRMSNorm

class MixtralAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MistralDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
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
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        
        # experts
        # MoE has multiple MLPs called experts
        for i, expert in enumerate(module.mlp.experts):
            # routed MLP in
            # TODO: figure out if prev_op can be removed
            if i == 0:
                prev_op = module.post_attention_layernorm
            else:
                prev_op = module.mlp.experts[i].w2
            
            # w1 = gate_proj
            # w2 = down_proj
            # w3 = up_proj
            # w2(F.silu(w1(x)) * w3(x))
            layers.append(dict(
                prev_op=prev_op,
                layers=[expert.w1, expert.w3],
                inp=input_feat[f'mlp.experts.{i}.w1'],
                module2inspect=module.mlp,
            ))

            # routed MLP out
            layers.append(dict(
                prev_op=expert.w3,
                layers=[expert.w2],
                inp=input_feat[f'mlp.experts.{i}.w2'],
            ))

        return layers


class MixtralFuser:
    def __init__(self, model: OldMistralForCausalLM):
        self.model = model

        self.mistral_blocks: List[Tuple[str, OldMistralDecoderLayer]] = [
            (name, module) for name, module in self.model.named_modules()
            if 'MistralDecoderLayer'.lower() in module.__class__.__name__.lower()
        ]
    
    def fuse_transformer(self):
        blocks = []

        module: OldMistralDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj
            )
            # Adapt to mixture of experts
            for i in range(len(module.experts)):
                mlp = QuantFusedMLP(
                    gate_proj=module.experts[i].w1,
                    down_proj=module.experts[i].w2,
                    up_proj=module.experts[i].w3
                )
                module.experts[i] = mlp
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight,
                module.input_layernorm.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon
            )
            blocks.append(MixtralBlock(
                hidden_size=self.model.config.hidden_size,
                n_heads=self.model.config.num_attention_heads,
                n_kv_heads=self.model.config.num_key_value_heads,
                qkv_layer=qkv,
                o_proj=module.self_attn.o_proj,
                moe=module.experts,
                norm_1=norm_1,
                norm_2=norm_2,
                dev=device,
                max_seq_len=self.model.config.max_new_tokens
            ))
        
        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )

