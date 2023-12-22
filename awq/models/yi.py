import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from awq.modules.fused.mlp import QuantFusedMLP
from awq.modules.fused.norm import FasterTransformerRMSNorm

class YiAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "YiDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model):
        fuser = YiFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.ln1,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.ln2,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))

        return layers


class YiFuser:
    def __init__(self, model):
        self.model = model

        self.yi_blocks: List[Tuple[str, object]] = [
            (name, module) for name, module in self.model.named_modules()
            if 'YiDecoderLayer'.lower() in module.__class__.__name__.lower()
        ]
    
    def fuse_transformer(self):
        blocks = []

        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj
            )
            mlp = QuantFusedMLP(
                module.mlp.gate_proj,
                module.mlp.down_proj,
                module.mlp.up_proj
            )
            norm_1 = FasterTransformerRMSNorm(
                module.ln1.weight,
                module.ln1.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.ln2.weight,
                module.ln2.variance_epsilon
            )
            blocks.append(LlamaLikeBlock(
                hidden_size=self.model.config.hidden_size,
                n_heads=self.model.config.num_attention_heads,
                n_kv_heads=self.model.config.num_key_value_heads,
                qkv_layer=qkv,
                o_proj=module.self_attn.o_proj,
                mlp=mlp,
                norm_1=norm_1,
                norm_2=norm_2,
                dev=device,
                max_seq_len=self.model.config.max_new_tokens,
                rope_theta=self.model.config.rope_theta
            ))
        
        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
