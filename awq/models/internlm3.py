import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM


class InternLM3AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "InternLM3DecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(
            is_scalable=True,
            scale_name="mlp.down_proj",
            scale_layer=module.mlp.down_proj,
            scale_shape=module.mlp.down_proj.out_features,
        )

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # attention input - separate Q, K, V projections in InternLM3
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.qkv"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        layers.append(
            dict(
                prev_op=module.self_attn.q_proj,  # Using q_proj as representative
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

        # feed forward input - gate_proj and up_proj in InternLM3
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[
                    module.mlp.gate_proj,
                    module.mlp.up_proj,
                ],
                inp=input_feat["mlp.gate_up_proj"],
                module2inspect=module.mlp,
                kwargs=module_kwargs,
            )
        )

        # feed forward output
        layers.append(
            dict(
                prev_op=module.mlp.gate_proj,  # Using gate_proj as representative
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers
