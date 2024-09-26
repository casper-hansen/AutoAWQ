import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM


class DeepseekAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "DeepseekDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(
        module, input_feat, module_kwargs
    ):
        layers = []
        
        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        
        if hasattr(module.mlp, "gate"):
            # linear in
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[
                        w
                        for expert in module.mlp.experts
                        for w in [expert.gate_proj, expert.up_proj]
                    ] + [module.mlp.shared_experts.gate_proj, module.mlp.shared_experts.up_proj],
                    inp=input_feat["mlp"],
                    module2inspect=module.mlp,
                )
            )

            # linear out
            for i, expert in enumerate(module.mlp.experts):
                layers.append(
                    dict(
                        prev_op=expert.up_proj,
                        layers=[expert.down_proj],
                        inp=input_feat[f"mlp.experts.{i}.down_proj"],
                    )
                )
            layers.append(
                dict(
                    prev_op=module.mlp.shared_experts.up_proj,
                    layers=[module.mlp.shared_experts.down_proj],
                    inp=input_feat[f"mlp.shared_experts.down_proj"],
                )
            )
        else:
            # linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
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
