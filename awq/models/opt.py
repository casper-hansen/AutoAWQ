from .base import BaseAWQForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTDecoderLayer


class OptAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "OPTDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model: OPTForCausalLM):
        return model.model.decoder.layers

    @staticmethod
    def get_act_for_scaling(module: OPTDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OPTForCausalLM, device: str):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )

    @staticmethod
    def fake_input_feat():
        return {
            "self_attn.q_proj": None,
            "self_attn.out_proj": None,
            "fc1": None,
            "fc2": None,
        }
        
    @staticmethod
    def get_layers_for_scaling(module: OPTDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.self_attn_layer_norm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
                layer_names=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                prev_op_name="self_attn_layer_norm",
            )
        )

        # attention out
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
                layer_names=["self_attn.out_proj"],
                prev_op_name="self_attn.v_proj",
            )
        )

        # linear 1
        layers.append(
            dict(
                prev_op=module.final_layer_norm,
                layers=[module.fc1],
                inp=input_feat["fc1"],
                layer_names=["fc1"],
                prev_op_name="final_layer_norm",
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.fc1,
                layers=[module.fc2],
                inp=input_feat["fc2"],
                layer_names=["fc2"],
                prev_op_name="fc1",
            )
        )

        return layers
