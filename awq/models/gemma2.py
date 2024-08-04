import copy
import tqdm
import torch
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import Gemma2LikeBlock
from awq.modules.fused.model import Gemma2LikeModel
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer as OldGemmaDecoderLayer,
    Gemma2ForCausalLM as OldGemmaForCausalLM,
)
from awq.modules.fused.norm import FasterTransformerRMSNorm


class Gemma2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Gemma2DecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldGemmaDecoderLayer):
        fuser = GemmaFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldGemmaForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldGemmaDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldGemmaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldGemmaDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
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

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        layers.append(
            dict(
                prev_op=module.pre_feedforward_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers

class GemmaFuser:
    def __init__(self, model: OldGemmaForCausalLM):
        self.model = model

        self.Gemma_blocks: List[Tuple[str, OldGemmaDecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "Gemma2DecoderLayer".lower() in module.__class__.__name__.lower() #Gemma2DecoderLayer
        ]

    def fuse_transformer(self):
        blocks = []
        
        module: OldGemmaDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            with torch.no_grad():
                # GemmaRMSNorm is different from Llama's in that it multiplies
                # (1 + weight) to the output, instead of just weight.
                module.input_layernorm.weight += 1
                module.post_attention_layernorm.weight += 1
                module.pre_feedforward_layernorm.weight += 1
                module.post_feedforward_layernorm.weight += 1

            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.eps
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.eps,
            )
            norm_3 = FasterTransformerRMSNorm(
                module.pre_feedforward_layernorm.weight,
                module.pre_feedforward_layernorm.eps
            )
            norm_4 = FasterTransformerRMSNorm(
                module.post_feedforward_layernorm.weight,
                module.post_feedforward_layernorm.eps,
            )
            blocks.append(
                Gemma2LikeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    norm_3=norm_3,
                    norm_4=norm_4,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                    head_dim=self.model.config.head_dim,
                    attn_logit_softcapping=self.model.config.attn_logit_softcapping,
                )
            )
        
        self.model.model = Gemma2LikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
            self.model.config.hidden_size,
        )
        
        setattr(self.model.model, "blocks", self.model.model.blocks)