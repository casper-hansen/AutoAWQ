import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from transformers.models.whisper.modeling_whisper import (
    WhisperDecoderLayer as OldWhisperDecoderLayer,
)
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration as OldWhisperForConditionalGeneration
from awq.modules.fused.mlp import QuantFusedMLP
from awq.modules.fused.norm import FasterTransformerRMSNorm

class WhisperAWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = "WhisperDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"
    is_encoder_decoder = True

    def get_input_embeds(self):
        return self.model.model.decoder.embed_tokens

    @staticmethod
    def fuse_layers(model: OldWhisperForConditionalGeneration):
        fuser = LlavaFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldWhisperForConditionalGeneration):
        return model.model.decoder.layers
    
    @staticmethod
    def get_act_for_scaling(module: OldWhisperDecoderLayer):
        return dict(
            is_scalable=False
        )

    @staticmethod
    def move_embed(model: OldWhisperForConditionalGeneration, device: str):
        model.proj_out = model.get_output_embeddings().to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: OldWhisperDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.self_attn_layer_norm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.out_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.out_proj],
                inp=input_feat['self_attn.out_proj'],
            ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.final_layer_norm,
            layers=[module.fc1],
            inp=input_feat['fc1'],
        ))

        layers.append(dict(
            prev_op=module.fc1,
            layers=[module.fc2],
            inp=input_feat['fc2'],
        ))

        return layers

