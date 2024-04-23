from .base import BaseAWQForCausalLM
from transformers.models.cohere.modeling_cohere import (
    CohereDecoderLayer as OldCohereDecoderLayer,
    CohereForCausalLM as OldCohereForCausalLM,
)


class CohereAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "CohereDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldCohereForCausalLM):
        pass

    @staticmethod
    def get_model_layers(model: OldCohereForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldCohereDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldCohereForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

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
