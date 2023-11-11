from .base import BaseAWQForCausalLM
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer as OldFalconDecoderLayer, FalconForCausalLM, FalconAttention

class FalconAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "FalconDecoderLayer"

    @staticmethod
    def fuse_layers(model: FalconForCausalLM):
        fuser = FalconFuser(model)

        # TODO: Implement correctly fused modules for Falcon 40B and Falcon 180B
        if model.config.num_attention_heads == 71:
            fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: FalconForCausalLM):
        return model.transformer.h
    
    @staticmethod
    def get_act_for_scaling(module: OldFalconDecoderLayer):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.dense_h_to_4h.out_features
        )
    
    @staticmethod
    def move_embed(model: FalconForCausalLM, device):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: OldFalconDecoderLayer, input_feat, module_kwargs):
        layers = []
        
        # Falcon 7B (older architecture)
        if module.config.num_attention_heads == 71:
            # linear 1 + attention
            layers.append(dict(
                prev_op=module.input_layernorm,
                layers=[module.mlp.dense_h_to_4h, module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

        # Falcon 40B (newer architecture)
        else:
            # linear 1 + attention
            layers.append(dict(
                prev_op=module.ln_attn,
                layers=[module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

            # linear 2
            layers.append(dict(
                prev_op=module.ln_mlp,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat['mlp.dense_h_to_4h'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

        return layers

from awq.modules.fused.model import FalconModel
from awq.modules.fused.block import FalconDecoderLayer

class FalconFuser:
    def __init__(self, model: FalconForCausalLM):
        self.model = model
    
    def fuse_transformer(self):
        blocks = []

        module: OldFalconDecoderLayer
        for module in self.model.transformer.h:
            if module.config.num_attention_heads == 71:
                input_layernorm = module.input_layernorm
                ln_attn = None
                ln_mlp = None
                new_decoder_arch = False
            else:
                input_layernorm = None
                ln_attn = module.ln_attn
                ln_mlp = module.ln_mlp
                new_decoder_arch = True
            
            blocks.append(FalconDecoderLayer(
                hidden_size=module.config.hidden_size,
                n_heads=module.config.num_attention_heads,
                qkv_layer=module.self_attention.query_key_value,
                o_proj=module.self_attention.dense,
                mlp=module.mlp,
                dev=next(iter(module.state_dict().values())).device,
                max_seq_len=self.model.config.max_new_tokens,
                input_layernorm=input_layernorm,
                ln_attn=ln_attn,
                ln_mlp=ln_mlp,
                new_decoder_arch=new_decoder_arch
            ))

        self.model.transformer = FalconModel(
            self.model.config.vocab_size,
            blocks,
            self.model.transformer.word_embeddings,
            self.model.transformer.ln_f,
        )