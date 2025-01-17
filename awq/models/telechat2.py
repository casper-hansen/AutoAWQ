from typing import List, Tuple
from .base import BaseAWQForCausalLM

# VLLM now supports telechat2,
# but the transformer library does not yet support it,
# so the following type prompts will be improved later

class OldTelechat2Block:
    """
    Just as a type description, consistent with the modeling_telechat.py
    """
    ...

class OldTelechatForCausalLM:
    """
    Just as a type description, consistent with the modeling_telechat.py
    """
    ...

class Telechat2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "TelechatBlock"
    max_seq_len_key = "training_seqlen"
    modules_to_not_convert = ['query', 'key_value', 'dense']

    @staticmethod
    def fuse_layers(model: OldTelechatForCausalLM):
        raise NotImplementedError()

    @staticmethod
    def get_model_layers(model: OldTelechatForCausalLM):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module: OldTelechat2Block):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldTelechatForCausalLM, device: str):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldTelechat2Block, input_feat, module_kwargs):
        """
        Due to the unique organization of key value weights in telechat2,
        the qkv and dense layers cannot be quantified
        """
        layers = []

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
                kwargs={
                    "residual": input_feat['post_attention_layernorm']
                }
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

