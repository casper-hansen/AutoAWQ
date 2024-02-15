from .base import BaseAWQForCausalLM


class QwenAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "QWenBlock"
    max_seq_len_key = "seq_length"

    @staticmethod
    def get_model_layers(model):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.rotary_emb = model.transformer.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # attention
        layers.append(
            dict(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat["attn.c_attn"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )

        # mlp
        layers.append(
            dict(
                prev_op=module.ln_2,
                layers=[module.mlp.w2, module.mlp.w1],
                inp=input_feat["mlp.w2"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.w1,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )

        return layers
