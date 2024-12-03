import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
try:
    from transformers.models.exaone.modeling_exaone import (
        ExaoneBlock as OldExaoneBlock,
        ExaoneForCausalLM as OldExaoneForCausalLM,
    )
except:
    OldExaoneBlock = None
    OldExaoneForCausalLM = None
from awq.modules.fused.norm import FasterTransformerRMSNorm


class ExaoneAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "ExaoneBlock"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldExaoneForCausalLM):
        fuser = LlamaFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldExaoneForCausalLM):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module: OldExaoneBlock):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldExaoneForCausalLM, device: str):
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.rotary = model.transformer.rotary.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldExaoneBlock, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.ln_1,
                layers=[
                    module.attn.attention.q_proj,
                    module.attn.attention.k_proj,
                    module.attn.attention.v_proj,
                ],
                inp=input_feat["attn.attention.q_proj"],
                module2inspect=module.attn.attention,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.attn.attention.v_proj.weight.shape == module.attn.attention.out_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.attn.attention.v_proj,
                    layers=[module.attn.attention.out_proj],
                    inp=input_feat["attn.attention.out_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.ln_2,
                layers=[module.mlp.c_fc_0, module.mlp.c_fc_1],
                inp=input_feat["mlp.c_fc_0"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.c_fc_1,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )

        return layers


class LlamaFuser:
    def __init__(self, model: OldExaoneForCausalLM):
        self.model = model

        self.llama_blocks: List[Tuple[str, OldExaoneBlock]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "LlamaDecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldExaoneBlock
        for module in tqdm.tqdm(self.model.transformer.h, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.attn.attention.q_proj,
                module.attn.attention.k_proj,
                module.attn.attention.v_proj,
            )
            norm_1 = FasterTransformerRMSNorm(
                module.ln_1.weight, module.ln_1.eps
            )
            norm_2 = FasterTransformerRMSNorm(
                module.ln_2.weight,
                module.ln_2.eps,
            )
            blocks.append(
                LlamaLikeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.attn.attention.out_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                )
            )

        self.model.transformer = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.transformer.wte,
            self.model.transformer.ln_f,
        )
        setattr(self.model.transformer, "blocks", self.model.transformer.blocks)
