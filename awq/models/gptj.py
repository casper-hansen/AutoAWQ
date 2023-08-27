from .base import BaseAWQForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM, GPTJBlock

class GPTJAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "GPTJBlock"
    max_new_tokens_key = "n_positions"

    @staticmethod
    def get_model_layers(model: GPTJForCausalLM):
        return model.transformer.h
    
    @staticmethod
    def get_act_for_scaling(module: GPTJBlock):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.fc_in.out_features
        )
    
    @staticmethod
    def move_embed(model: GPTJForCausalLM, device: str):
        model.transformer.wte = model.transformer.wte.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: GPTJBlock, input_feat, module_kwargs):
        layers = []

        # attention input + linear 1
        layers.append(dict(
            prev_op=module.ln_1,
            layers=[module.attn.q_proj,
                    module.attn.k_proj, module.attn.v_proj, module.mlp.fc_in],
            inp=input_feat['attn.q_proj'],
            module2inspect=module,
            kwargs=module_kwargs
        ))

        # attention out
        layers.append(dict(
            prev_op=module.attn.v_proj,
            layers=[module.attn.out_proj],
            inp=input_feat['attn.out_proj'],
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.act,
            layers=[module.mlp.fc_out],
            inp=input_feat['mlp.fc_out'],
        ))

        return layers