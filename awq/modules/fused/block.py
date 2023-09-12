import torch.nn as nn
from awq.modules.fused.attn import QuantAttentionFused

class MPTBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, qkv_layer, o_proj, mpt_mlp, norm_1, norm_2, dev, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.norm_1 = norm_1
        self.attn = QuantAttentionFused(hidden_size, self.n_heads, qkv_layer, o_proj, dev=dev, max_seq_len=max_seq_len, use_alibi=True).to(dev)
        self.norm_2 = norm_2
        self.ffn = mpt_mlp.to(dev)

    def forward(
        self, hidden_states, past_key_value, attn_bias=None, attention_mask=None, is_causal=None
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=norm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=False,
            use_cache=True
        )

        h = hidden_states + attn_output
        out = h + self.ffn.forward(self.norm_2(h))
        return out, None, past_key_value

class FalconDecoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, qkv_layer, o_proj, mlp, dev, max_seq_len, input_layernorm=None, ln_attn=None, ln_mlp=None, new_decoder_arch=True):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        # TODO: Falcon has ALiBi implemented but which model uses it?
        self.attn = QuantAttentionFused(hidden_size, self.n_heads, qkv_layer, o_proj, dev=dev, max_seq_len=max_seq_len, use_alibi=False).to(dev)
        self.new_decoder_arch = new_decoder_arch
        
        if new_decoder_arch:
            self.ln_attn = ln_attn # before attention
            self.ln_mlp = ln_mlp # before mlp
        else:
            self.input_layernorm = input_layernorm # before attention
        
        self.mlp = mlp

    def forward(
        self, hidden_states, past_key_value, attn_bias=None, attention_mask=None, is_causal=None
    ):
        if self.new_decoder_arch:
            layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            layernorm_out = self.input_layernorm(hidden_states)
        
        attn_output, _, past_key_value = self.attn.forward(
            hidden_states=layernorm_out,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=None,
            output_attentions=False,
            use_cache=True
        )

        h_attn = hidden_states + attn_output

        if self.new_decoder_arch:
            h_mlp = self.mlp.forward(mlp_layernorm_out)
        else:
            h_mlp = self.mlp.forward(layernorm_out)
        
        out = h_attn + h_mlp
        
        return out, None, past_key_value
