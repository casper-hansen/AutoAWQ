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