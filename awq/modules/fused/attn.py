import math
import torch
import torch.nn as nn
import awq_inference_engine
from torch.nn import functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_INSTALLED = True
except:
    FLASH_INSTALLED = False

class QuantLlamaRotary(nn.Module):
    def __init__(self, dim=4096, max_position_embeddings=2048, base=10000, device=None, 
                       is_neox=True, num_heads=None, num_kv_heads=None):
        super().__init__()
        self.dim = dim
        self.is_neox = is_neox
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # create cache
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2, device=device) / dim))
        t = torch.arange(max_position_embeddings, device=device).float()
        freqs = torch.einsum("i,j -> ij", t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).to(torch.get_default_dtype())

        # Embedding size: [max_position, rotary_dim]
        self.register_buffer("cos_sin_cache", cache.half(), persistent=False)
    
    def forward(
        self,
        qkv_states: torch.Tensor,
        position_ids: torch.Tensor,
        batch_size: int, 
        q_len: int
    ):
        # get qkv
        query, key, value = qkv_states.chunk(chunks=3, dim=-1)
        del qkv_states

        # [num_tokens, num_heads * head_size]
        query_batch_size, query_len, _ = query.shape
        query = query.view(query_len*query_batch_size, self.num_heads * self.dim)

        # [num_tokens, num_kv_heads * head_size]
        key_batch_size, key_len, _ = key.shape
        key = key.view(key_len*key_batch_size, self.num_kv_heads * self.dim)

        # [num_tokens]
        positions = position_ids.view(-1).to(query.device)

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
            self.is_neox
        )

        query = query.view(batch_size, q_len, self.num_heads, self.dim).transpose(1, 2)
        key = key.view(batch_size, q_len, self.num_heads, self.dim).transpose(1, 2)
        value = value.view(batch_size, q_len, self.num_heads, self.dim).transpose(1, 2)

        return query, key, value


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

        # self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        # self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
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
        awq_inference_engine.rotary_embedding(
            positions,
            query,
            key,
            self.dim,
            self.cos_sin_cache,
            True
        )
        return query, key


class TorchAttention(nn.Module):
    def __init__(self, hidden_size, use_flash=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_flash = use_flash
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        use_cache: bool,
        past_key_value: torch.Tensor,
        batch_size: int, 
        q_len: int
    ):
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

        if self.use_flash and FLASH_INSTALLED:
            query = query.transpose(1,2)
            key = key.transpose(1,2)
            value = value.transpose(1,2)
            attn_output = flash_attn_func(query, key, value, causal=is_causal)
        else:
            attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        
        del query, key, value

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q_len, self.hidden_size)

        return attn_output, past_key_value

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
        max_new_tokens
    ):
        super().__init__()
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.attn = TorchAttention(hidden_size)

        self.rotary_emb = QuantLlamaRotary(
            dim=hidden_size // num_heads,
            max_position_embeddings=max_new_tokens,
            device=dev,
            is_neox=True,
            num_heads=num_heads,
            num_kv_heads=num_heads
        )


    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False):
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()
        qkv_states = self.qkv_proj(hidden_states)
        query, key, value = self.rotary_emb(qkv_states, position_ids, batch_size, q_len)
        attn_output, past_key_value = self.attn(query, key, value, use_cache, past_key_value, batch_size, q_len)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

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


class QuantLlamaAttentionFused(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_layer, o_proj, dev, max_position_embeddings):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = self.hidden_size // num_heads
        self.qkv_proj = qkv_layer
        self.o_proj = o_proj
        self.start_pos = 0

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim ,
            max_position_embeddings * 2,
        )

        # following fastertransformer definition
        self.cache_v = (
            torch.zeros(
                (
                    1,
                    self.n_local_heads,
                    max_position_embeddings,
                    self.head_dim,
                )
            )
            .to(dev)
            .half()
        )  # added to half
        # 8: pack 8 fp16 in FT, if fp32 then use 4
        self.cache_k = (
            torch.zeros(
                (
                    1,
                    self.n_local_heads,
                    self.head_dim // 8,
                    max_position_embeddings,
                    8,
                )
            )
            .to(dev)
            .half()
        )  # added to half

        # dummy
        self.rotary_emb = QuantLlamaRotaryEmbedding(
            hidden_size // num_heads, max_position_embeddings=max_position_embeddings, base=10000, device=dev
        )
        
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

            xq, xk = self.rotary_emb(xq, xk, position_ids)

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

            keys = xk
            values = xv
            past_key_value = (xk, xv) if use_cache else None

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores + attention_mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
            xq = xq[:, 0, :, :]
            xk = xk[:, 0, :, :]
            xv = xv[:, 0, :, :]
            past_key_value = (xk, xv) if use_cache else None
            output = awq_inference_engine.single_query_attention(
                xq,
                xk,
                xv,
                self.cache_k,
                self.cache_v,
                None,
                None,
                self.start_pos,
                self.head_dim,
                10000,
                True,
            )
            output = output.reshape(bsz, 1, -1)
        
        attn_output = self.o_proj(output)
        
        if use_cache:
            self.start_pos += seqlen
        else:
            self.start_pos = 0

        return attn_output, None, past_key_value
