import logging
from .base import BaseAWQForCausalLM

try:
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralForCausalLM
except:
    # TODO: Remove once released on PyPi
    logging.warning("You need the latest transformers 4.34.0.dev0: pip install git+https://github.com/huggingface/transformers.git")
    MistralForCausalLM = None
    MistralDecoderLayer = None

class MistralAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MistralDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model: MistralForCausalLM):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module: MistralDecoderLayer):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model: MistralForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: MistralDecoderLayer, input_feat, module_kwargs):
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
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))

        return layers