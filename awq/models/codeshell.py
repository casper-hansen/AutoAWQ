from .base import BaseAWQForCausalLM

class CodeShellAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "CodeShellBlock"
    max_new_tokens_key = "n_positions"

    @staticmethod
    def get_model_layers(model):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.c_fc.out_features
        )

    @staticmethod
    def move_embed(model, device: str):
        model.transformer.wte = model.transformer.wte.to(device)


    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []
        # attention
        layers.append(
            dict(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat['attn.c_attn'],
            )
        )
        # mlp
        layers.append(
            dict(
                prev_op=module.ln_2,
                layers=[module.mlp.c_fc],
                inp=input_feat['mlp.c_fc']
            )
        )
        layers.append(
            dict(
                prev_op=module.mlp.act,
                layers=[module.mlp.c_proj],
                inp=input_feat['mlp.c_proj']
            )
        )
        return layers