import math
import torch
import torch.nn as nn
import awq_inference_engine
from torch.nn import functional as F

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
    )
    xk_ = torch.view_as_complex(
        xk.float().reshape(*xk.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-2, -1).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-2, -1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def gen_slopes(n_heads, alibi_bias_max=8):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(
    n_heads, seq_len, full=False, alibi_bias_max=8, dtype=torch.float32
):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32).view(1, 1, 1, seq_len)
    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32).view(
            1, 1, seq_len, 1
        )
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max)
    alibi_bias = alibi_bias * slopes
    slopes = slopes.squeeze(0).squeeze(-1).squeeze(-1)
    return slopes.to(dtype=dtype), alibi_bias.to(dtype=dtype)


class QuantLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        self.register_buffer("cos_sin_cache", cache.half(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ):
        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        # print(positions.shape, query.shape, key.shape, self.cos_sin_cache.shape)
        query = query.contiguous()
        key = key.contiguous()
        awq_inference_engine.rotary_embedding_neox(
            positions,
            query,
            key,
            self.dim,
            self.cos_sin_cache
        )
        return query, key

class QuantAttentionFused(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_layer, o_proj, dev, max_seq_len, use_alibi=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = self.hidden_size // num_heads
        self.qkv_proj = qkv_layer
        self.o_proj = o_proj
        self.start_pos = 0
        self.use_alibi = use_alibi
        self.cache_batch_size = 1

        # following fastertransformer definition
        self.cache_v = (
            torch.zeros(
                ( self.cache_batch_size, self.n_local_heads, max_seq_len, self.head_dim, )
            ).to(dev).half()
        )
        
        # 8: pack 8 fp16 in FT, if fp32 then use 4
        self.cache_k = (
            torch.zeros(
                ( self.cache_batch_size, self.n_local_heads, self.head_dim // 8, max_seq_len, 8, )
            ).to(dev).half()
        )

        if use_alibi:
            alibi_slopes, alibi_bias = build_alibi_bias(self.n_local_heads, max_seq_len)
            self.alibi_slopes = alibi_slopes.float().to(dev)
            self.alibi_bias = alibi_bias.float().to(dev)
            self.rotary_dim = 0
            self.is_neox = False
        else:
            self.freqs_cis = precompute_freqs_cis(
                hidden_size // num_heads,
                max_seq_len * 2,
            ).to(dev)
            self.rotary_dim = self.head_dim
            self.alibi_slopes = None
            self.is_neox = True
    
    def _multi_query_attention_torch(self, query, key, value, batch_size, seqlen, use_cache, past_key_value, attention_mask):
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            key = key.contiguous()
            value = value.contiguous()
            query = query.contiguous()

        output = F.scaled_dot_product_attention(
            query, key, value, 
            is_causal=past_key_value is None, 
            attn_mask=attention_mask)
        

        del query, key, value

        output = output.transpose(1, 2).reshape(batch_size, seqlen, self.hidden_size)

        return output
    
    def forward(
        self,
        hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False
    ):
        bsz, seqlen, _ = hidden_states.shape
        xqkv = self.qkv_proj(hidden_states)
        xqkv = xqkv.view(bsz, seqlen, -1, self.n_local_heads, self.head_dim)
        xq = xqkv[:, :, 0]
        xk = xqkv[:, :, 1]
        xv = xqkv[:, :, 2]

        if seqlen > 1:
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

            if not self.use_alibi:
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis[self.start_pos : self.start_pos + seqlen])

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape(bsz, seqlen, self.n_local_heads, self.head_dim // 8, 8)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )

            self.cache_v[:bsz, :, self.start_pos : self.start_pos + seqlen, :] = values_store
            self.cache_k[:bsz, :, :, self.start_pos : self.start_pos + seqlen, :] = keys_store

            past_key_value = (xk, xv) if use_cache else None
            output = self._multi_query_attention_torch(
                xq, xk, xv, 
                bsz, seqlen, True, 
                past_key_value, attention_mask
            )
        else:
            xq = xq[:, 0, :, :]
            xk = xk[:, 0, :, :]
            xv = xv[:, 0, :, :]
            past_key_value = (xk, xv) if use_cache else None
            output = awq_inference_engine.single_query_attention(
                xq, # query
                xk, # key
                xv, # value
                self.cache_k, # key cache
                self.cache_v, # value cache
                None, # length per sample
                self.alibi_slopes, # alibi slopes
                self.start_pos, # timestep
                self.rotary_dim, # rotary embedding dimension
                10000, # rotary embedding base
                self.is_neox, # is neox
            )
            output = output.reshape(bsz, 1, -1)
        
        attn_output = self.o_proj(output)
        
        if use_cache:
            self.start_pos += seqlen
        else:
            self.start_pos = 0

        return attn_output, None, past_key_value