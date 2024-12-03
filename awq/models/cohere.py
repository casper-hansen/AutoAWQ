import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import CohereBlock
from awq.modules.fused.model import CohereModel
from transformers.models.cohere.modeling_cohere import (
    CohereDecoderLayer as OldCohereDecoderLayer,
    CohereForCausalLM as OldCohereForCausalLM,
)
from awq.modules.fused.norm import FasterTransformerRMSNorm

class CohereAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "CohereDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldCohereForCausalLM):
        fuser = CohereFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldCohereForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldCohereDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldCohereForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(
        module: OldCohereDecoderLayer, input_feat, module_kwargs
    ):
        layers = []

        # input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                    module.mlp.gate_proj,
                    module.mlp.up_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module,
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

        # linear out
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers

class CohereFuser:
    def __init__(self, model: OldCohereForCausalLM):
        self.model = model

        self.cohere_blocks: List[Tuple[str, OldCohereDecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "CohereDecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldCohereDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            norm_1 = module.input_layernorm
            # norm_2 = FasterTransformerRMSNorm(
            #     module.post_attention_layernorm.weight,
            #     module.post_attention_layernorm.variance_epsilon,
            # )
            blocks.append(
                CohereBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    # norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                )
            )

        self.model.model = CohereModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
