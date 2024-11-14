import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM


class InternLM2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "InternLM2DecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(
            is_scalable=True,
            scale_name="feed_forward.w2",
            scale_layer=module.feed_forward.w2,
            scale_shape=module.feed_forward.w2.out_features,
        )

    @staticmethod
    def move_embed(model, device: str):
        model.model.tok_embeddings = model.model.tok_embeddings.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.attention_norm,
                layers=[
                    module.attention.wqkv,
                ],
                inp=input_feat["attention.wqkv"],
                module2inspect=module.attention,
                kwargs=module_kwargs,
            )
        )

        # attention out
        layers.append(
            dict(
                prev_op=module.attention.wqkv,
                layers=[module.attention.wo],
                inp=input_feat["attention.wo"],
            )
        )

        # feed forward input
        layers.append(
            dict(
                prev_op=module.ffn_norm,
                layers=[
                    module.feed_forward.w1,
                    module.feed_forward.w3,
                ],
                inp=input_feat["feed_forward.w1"],
                module2inspect=module.feed_forward,
                kwargs=module_kwargs,
            )
        )

        # feed forward output
        layers.append(
            dict(
                prev_op=module.feed_forward.w1,
                layers=[module.feed_forward.w2],
                inp=input_feat["feed_forward.w2"],
            )
        )

        return layers
