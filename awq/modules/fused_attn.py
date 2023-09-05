import torch
import torch.nn as nn
import awq_inference_engine
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaLinearScalingRotaryEmbedding

class RotaryEmbeddingNeox(nn.Module):
    def __init__(self, head_dim, seq_len, device):
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.base = 10000

        # create inv_frequency
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # set cache
        self._set_cos_sin_cache(seq_len=self.seq_len, device=self.inv_freq.device)
    
    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        
        self.register_buffer("cos_sin_cache", cache.half(), persistent=False)
    
    def forward(self, positions, query, key):
        batch_size, seq_len, _ = query.shape
        query = query.view(batch_size * seq_len, -1)
        key = key.view(batch_size * seq_len, -1)
        positions = positions.view(-1).to(query.device)

        query = query.contiguous()
        key = key.contiguous()

        awq_inference_engine.rotary_embedding_neox(
            positions,
            query,
            key,
            self.head_dim,
            self.cos_sin_cache,
        )
        query = query.view(batch_size, seq_len, -1)
        key = key.view(batch_size, seq_len, -1)

        return query, key
    
class QuantLlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        qkv_proj,
        o_proj,
        device,
        max_new_tokens
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.seq_len = max_new_tokens
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.rotary_embedding_neox = RotaryEmbeddingNeox(self.head_dim, self.seq_len, device)

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {num_heads}).")

    def attn(self, query, key, value, past_key_value, use_cache, attention_mask):
        batch_size, seq_len, _ = query.shape

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        value = value.to(key.device)

        # cache ops
        is_causal = past_key_value is None
        if past_key_value is not None:
            # reuse k, v, self_attention
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        if use_cache:
            # Since qkv_proj is fused, query_states etc will hold a reference to the original qkv tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        past_key_value = (key, value) if use_cache else None

        # multi-head masked attention
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None if is_causal else attention_mask,
            is_causal=is_causal
        )

        # reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        return attn_output, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # qkv proj
        query, key, value = self.qkv_proj(hidden_states).chunk(chunks=3, dim=-1)

        # rotary embeddings
        query, key = self.rotary_embedding_neox(position_ids, query, key)

        # attention
        attn_output, past_key_value = self.attn(query, key, value, past_key_value, use_cache, attention_mask)

        # out projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
