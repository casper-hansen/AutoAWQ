import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import PhiBlock
from awq.modules.fused.model import PhiModel as AWQPhiModel
from transformers.models.phi.modeling_phi import (
    PhiDecoderLayer as OldPhiDecoderLayer,
    PhiForCausalLM as OldPhiForCausalLM,
)
from awq.modules.fused.norm import FasterTransformerRMSNorm



class PhiAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "PhiDecoderLayer"
    max_seq_len_key = "max_position_embeddings"


    @staticmethod
    def fuse_layers(model: OldPhiForCausalLM):
        fuser = PhiFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldPhiForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldPhiForCausalLM):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldPhiForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldPhiDecoderLayer, input_feat, module_kwargs):
        layers = []

        #Attention:

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

        # Similarly to llama and other models, we skip the output projection
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696        
        if module.self_attn.v_proj.weight.shape == module.self_attn.dense.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.dense],
                    inp=input_feat["self_attn.dense"],
                )
            )

        # MLP:

        # linear 1
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[module.mlp.fc1],
                inp=input_feat["mlp.fc1"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.fc1,
                layers=[module.mlp.fc2],
                inp=input_feat["mlp.fc2"],
            )
        )

        return layers

class PhiFuser:
    def __init__(self, model: OldPhiForCausalLM):
        self.model = model

        self.phi_blocks: List[Tuple[str, OldPhiDecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "PhiDecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldPhiDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.eps
            )
            blocks.append(
                PhiBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    dense=module.self_attn.dense,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    dev=device,
                    max_seq_len=self.model.config.max_position_embeddings,
                    rope_theta=self.model.config.rope_theta,
                    rope_scaling=self.model.config.rope_scaling
                )
            )

        self.model.model = AWQPhiModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.final_layernorm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
