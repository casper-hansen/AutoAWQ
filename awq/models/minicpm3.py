from .base import BaseAWQForCausalLM

class MiniCPM3AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MiniCPMDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model):
        print(model.model.layers)
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # mlp
        layers.append(
            dict(
                prev_op=module.self_attn.q_a_layernorm,
                layers=[
                    module.self_attn.q_b_proj,
                    
                ],
                inp=input_feat["self_attn.q_b_proj"],
                module2inspect=module.self_attn.q_b_proj,
                kwargs=module_kwargs,
            )
        )

        layers.append(
            dict(
                prev_op=module.self_attn.kv_a_layernorm,
                layers=[
                    module.self_attn.kv_b_proj,
                ],
                inp=input_feat["self_attn.kv_b_proj"],
                module2inspect=module.self_attn.kv_b_proj,
                kwargs=module_kwargs,
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

        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj,module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp
            )
        )

        return layers