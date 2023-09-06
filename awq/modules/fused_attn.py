import torch
import torch.nn as nn
import awq_inference_engine
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding

class QuantLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        
        # [max_position, rot_dim]
        self.register_buffer("cos_sin_cache", cache.half(), persistent=False)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ):
        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        query = query.contiguous()
        key = key.contiguous()

        awq_inference_engine.rotary_embedding(
            positions,
            query,
            key,
            self.dim,
            self.cos_sin_cache,
            True # is_neox
        )

        return query, key

class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        qkv_proj,
        o_proj,
        dev,
        max_new_tokens,
        use_hf_rotary=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.use_hf_rotary = use_hf_rotary

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             f" and `num_heads`: {num_heads}).")
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj

        if use_hf_rotary:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_new_tokens, device=dev)
        else:
            self.rotary_emb = QuantLlamaRotaryEmbedding(self.head_dim, max_position_embeddings=max_new_tokens, device = dev)

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False):
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)

        if self.use_hf_rotary:
            # get qkv
            qkv_states = qkv_states.view(bsz, q_len, 3, self.num_heads, self.head_dim)
            query, key, value = torch.split(qkv_states, 1, dim=2)
            del qkv_states
            
            # reshape for hf rotary
            query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            
            cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        else:
            # get qkv
            query, key, value = qkv_states.chunk(chunks=3, dim=-1)
            del qkv_states

            # [num_tokens, num_heads * head_size]
            query_batch_size, query_len, _ = query.shape
            query = query.view(query_len*query_batch_size, self.num_heads * self.head_dim)

            # [num_tokens, num_kv_heads * head_size]
            key_batch_size, key_len, _ = key.shape
            key = key.view(key_len*key_batch_size, self.num_kv_heads * self.head_dim)

            # [num_tokens]
            positions = position_ids.view(-1).to(query.device)

            query, key = self.rotary_emb(query, key, positions)

            query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        is_causal = past_key_value is None

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        value = value.to(key.device)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        if use_cache:
            # Since qkv_proj is fused, query etc will hold a reference to the original qkv_states tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            key = key.contiguous()
            value = value.contiguous()
            query = query.contiguous()

        past_key_value = (key, value) if use_cache else None

        # with torch.backends.cuda.sdp_kernel(enable_math=False):
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        del query, key, value

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
