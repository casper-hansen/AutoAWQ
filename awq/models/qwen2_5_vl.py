from .base import BaseAWQForCausalLM
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer,
    )


class Qwen2_5_VLAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Qwen2_5_VLDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual"]

    @staticmethod
    def get_model_layers(model: "Qwen2_5_VLForConditionalGeneration"):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: "Qwen2_5_VLForConditionalGeneration"):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: "Qwen2_5_VLForConditionalGeneration", device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.visual = model.visual.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(
        module: "Qwen2_5_VLDecoderLayer", input_feat, module_kwargs
    ):
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

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers
