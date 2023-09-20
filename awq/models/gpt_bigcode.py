from .base import BaseAWQForCausalLM
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM, GPTBigCodeBlock as OldGptBigCodeBlock

class GptBigCodeAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "GPTBigCodeBlock"
    max_new_tokens_key = "n_positions"

    @staticmethod
    def fuse_layers(model: GPTBigCodeForCausalLM, quant_config:dict):
        # TODO: Fix single_query_attention
        pass
        # fuser = GptBigCodeFuser(model)
        # fuser.fuse_transformer()


    @staticmethod
    def get_model_layers(model: GPTBigCodeForCausalLM):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module: OldGptBigCodeBlock):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.c_fc.out_features
        )

    @staticmethod
    def move_embed(model: GPTBigCodeForCausalLM, device):
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.drop = model.transformer.drop.to(device)

    @staticmethod
    def get_layers_for_scaling(module:OldGptBigCodeBlock, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.ln_1,
            layers=[module.attn.c_attn],
            inp=input_feat['attn.c_attn'],
            module2inspect=module.attn,
            kwargs=module_kwargs
        ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.ln_2,
            layers=[module.mlp.c_fc],
            inp=input_feat['mlp.c_fc'],
            module2inspect=module.mlp
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.act,
            layers=[module.mlp.c_proj],
            inp=input_feat['mlp.c_proj']
        ))

        return layers

from typing import List, Tuple
from awq.modules.fused.block import GptBigCodeBlock
from awq.modules.fused.model import GptBigCodeModel

class GptBigCodeFuser:
    def __init__(self, model: GPTBigCodeForCausalLM):
        self.model = model

        self.blocks: List[Tuple[str, OldGptBigCodeBlock]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, OldGptBigCodeBlock)
        ]
    
    def fuse_transformer(self):
        blocks = []

        module: OldGptBigCodeBlock
        for module in self.model.transformer.h:
            blocks.append(GptBigCodeBlock(
                self.model.config.n_embd,
                self.model.config.n_head,
                module.attn.c_attn,
                module.attn.c_proj,
                module.mlp,
                module.ln_1,
                module.ln_2,
                next(iter(module.state_dict().values())).device, 
                self.model.config.n_positions
            ))

        self.model.transformer = GptBigCodeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.transformer.wte,
            self.model.transformer.wpe,
            self.model.transformer.ln_f,
        )
