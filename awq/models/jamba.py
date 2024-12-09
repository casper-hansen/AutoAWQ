import tqdm
import torch
from typing import List, Tuple, Union
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from transformers.models.jamba.modeling_jamba import (
    JambaAttentionDecoderLayer as OldJambaAttentionDecoderLayer,
    JambaMambaDecoderLayer as OldJambaMambaDecoderLayer,
    JambaForCausalLM as OldJambaForCausalLM,
)
from awq.modules.fused.norm import FasterTransformerRMSNorm


class JambaAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = ["JambaAttentionDecoderLayer", "JambaMambaDecoderLayer"]
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["mamba", "router"]

    @staticmethod
    def get_model_layers(model: OldJambaForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: Union[OldJambaMambaDecoderLayer, OldJambaAttentionDecoderLayer]):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldJambaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: Union[OldJambaMambaDecoderLayer, OldJambaAttentionDecoderLayer], input_feat, module_kwargs):
        layers = []

        # attention input
        if isinstance(module, OldJambaAttentionDecoderLayer):
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

            # attention out
            # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

        if hasattr(module.feed_forward, "router"):
            # linear in
            layers.append(
                dict(
                    prev_op=module.pre_ff_layernorm,
                    layers=[
                        w
                        for expert in module.feed_forward.experts
                        for w in [expert.gate_proj, expert.up_proj]
                    ],
                    inp=input_feat["feed_forward"],
                    module2inspect=module.feed_forward,
                )
            )

            # linear out
            for i, expert in enumerate(module.feed_forward.experts):
                layers.append(
                    dict(
                        prev_op=expert.up_proj,
                        layers=[expert.down_proj],
                        inp=input_feat[f"feed_forward.experts.{i}.down_proj"],
                    )
                )

        else:
            # linear 1
            layers.append(
                dict(
                    prev_op=module.pre_ff_layernorm,
                    layers=[module.feed_forward.gate_proj, module.feed_forward.up_proj],
                    inp=input_feat["feed_forward.gate_proj"],
                    module2inspect=module.feed_forward,
                )
            )

            # linear 2
            layers.append(
                dict(
                    prev_op=module.feed_forward.up_proj,
                    layers=[module.feed_forward.down_proj],
                    inp=input_feat["feed_forward.down_proj"],
                )
            )

        return layers
