import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2ForCausalLM as OldStarcoder2ForCausalLM,
    Starcoder2DecoderLayer as OldStarcoder2DecoderLayer,
)
from awq.modules.fused.norm import FasterTransformerRMSNorm


class Starcoder2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Starcoder2DecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldStarcoder2ForCausalLM):
        fuser = Starcoder2Fuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldStarcoder2ForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldStarcoder2DecoderLayer):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.c_fc.out_features,
        )
        # return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldStarcoder2ForCausalLM, device):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldStarcoder2DecoderLayer, input_feat, module_kwargs):
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
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.c_fc],
                inp=input_feat["mlp.c_fc"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.act,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )

        return layers

class Starcoder2Fuser:
    def __init__(self, model: OldStarcoder2ForCausalLM):
        self.model = model

        self.starcoder2_blocks: List[Tuple[str, OldStarcoder2DecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "Starcoder2DecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldStarcoder2DecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            # SC2 use normal LayerNorm
            norm_1 = module.input_layernorm
            norm_2 = module.post_attention_layernorm
            blocks.append(
                LlamaLikeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                )
            )

        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)