import tqdm
import torch
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv, fuse_w1w3
from awq.modules.fused.block import MixtralBlock
from awq.modules.fused.model import MixtralModel
from awq.modules.fused.moe import FusedSparseMoeBlock
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer as OldMixtralDecoderLayer,
    MixtralForCausalLM as OldMixtralForCausalLM
)
from awq.modules.fused.norm import FasterTransformerRMSNorm

class MixtralAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MixtralDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["gate"]
    
    @staticmethod
    def fuse_layers(model: OldMixtralForCausalLM):
        fuser = MixtralFuser(model)
        fuser.fuse_transformer()
    
    @staticmethod
    def get_model_layers(model: OldMixtralForCausalLM):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model: OldMixtralForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: OldMixtralDecoderLayer, input_feat, module_kwargs):
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
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        
        # linear in
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[
                w for expert in module.block_sparse_moe.experts
                  for w in [expert.w1, expert.w3]
            ],
            inp=input_feat['block_sparse_moe'],
            module2inspect=module.block_sparse_moe,
        ))

        # linear out
        for i, expert in enumerate(module.block_sparse_moe.experts):
            layers.append(dict(
                prev_op=expert.w3,
                layers=[expert.w2],
                inp=input_feat[f'block_sparse_moe.experts.{i}.w2'],
            ))

        return layers


class MixtralFuser:
    def __init__(self, model: OldMixtralForCausalLM):
        self.model = model

        self.mixtral_blocks: List[Tuple[str, OldMixtralDecoderLayer]] = [
            (name, module) for name, module in self.model.named_modules()
            if 'MixtralDecoderLayer'.lower() in module.__class__.__name__.lower()
        ]
    
    def fuse_transformer(self):
        blocks = []

        module: OldMixtralDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight,
                module.input_layernorm.variance_epsilon
            )

            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon
            )

            with torch.no_grad():
                fused_linears = [
                    fuse_w1w3(
                        module.block_sparse_moe.experts[i].w1,
                        module.block_sparse_moe.experts[i].w3,
                        device,
                    )
                    for i in range(len(module.block_sparse_moe.experts))
                ]

                from awq.modules.linear import WQLinear_GEMM

                w1w3 = WQLinear_GEMM(
                    fused_linears[0].w_bit,
                    fused_linears[0].group_size,
                    fused_linears[0].in_features,
                    fused_linears[0].out_features,
                    bias=None,
                    dev=device,
                )

                w1w3.qweight = torch.stack([w1w3.qweight for w1w3 in fused_linears], dim=0)
                w1w3.qzeros = torch.stack([w1w3.qzeros for w1w3 in fused_linears], dim=0)
                w1w3.scales = torch.stack([w1w3.scales for w1w3 in fused_linears], dim=0)

                w2s = WQLinear_GEMM(
                    module.block_sparse_moe.experts[0].w2.w_bit,
                    module.block_sparse_moe.experts[0].w2.group_size,
                    module.block_sparse_moe.experts[0].w2.in_features,
                    module.block_sparse_moe.experts[0].w2.out_features,
                    bias=None,
                    dev=device,
                )

                w2s.qweight = torch.stack([expert.w2.qweight for expert in module.block_sparse_moe.experts], dim=0)
                w2s.qzeros = torch.stack([expert.w2.qzeros for expert in module.block_sparse_moe.experts], dim=0)
                w2s.scales = torch.stack([expert.w2.scales for expert in module.block_sparse_moe.experts], dim=0)

                fused_block_sparse_moe = FusedSparseMoeBlock(
                    top_k=module.block_sparse_moe.top_k,
                    gate=module.block_sparse_moe.gate,
                    ws=w1w3,
                    w2s=w2s,
                )
                del module.block_sparse_moe.experts
            
            blocks.append(MixtralBlock(
                hidden_size=self.model.config.hidden_size,
                n_heads=self.model.config.num_attention_heads,
                n_kv_heads=self.model.config.num_key_value_heads,
                qkv_layer=qkv,
                o_proj=module.self_attn.o_proj,
                moe=fused_block_sparse_moe,
                norm_1=norm_1,
                norm_2=norm_2,
                dev=device,
                max_seq_len=self.model.config.max_seq_len,
                rope_theta=self.model.config.rope_theta
            ))
        
        self.model.model = MixtralModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)

