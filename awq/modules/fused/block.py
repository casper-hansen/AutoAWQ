import torch.nn as nn
from awq.modules.fused.attn import QuantAttentionFused

class MptBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, qkv_layer, o_proj, mpt_mlp, dev):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.attn = QuantAttentionFused(hidden_size, self.n_heads, qkv_layer, o_proj, dev="cuda:0", max_seq_len=8096, use_alibi=True)
        self.ffn = mpt_mlp
        self.norm_1 = nn.LayerNorm(hidden_size, eps=1e-6).half().to(dev)
        self.norm_2 = nn.LayerNorm(hidden_size, eps=1e-6).half().to(dev)

    def forward(
        self, hidden_states, past_key_value, attn_bias, attention_mask, is_causal
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