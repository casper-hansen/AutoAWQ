import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from awq.modules.fused.cache import WindowedCache
from awq.utils.fused_utils import get_attention_shapes


try:
    import awq_ft_ext
    FT_INSTALLED = True
except:
    FT_INSTALLED = False

HF_NEW_CACHE_FORMAT = False

import transformers
# https://github.com/huggingface/transformers/pull/26681 introduced a new cache format
HF_NEW_CACHE_FORMAT = hasattr(transformers, "cache_utils")
if HF_NEW_CACHE_FORMAT:
    from transformers.cache_utils import DynamicCache


class RoPE(nn.Module):
    def __init__(self, hidden_size, n_heads, max_seq_len, device, rope_theta):
        super(RoPE, self).__init__()
        
        self.freqs_cis = nn.Parameter(
            self.precompute_freqs_cis(
                hidden_size // n_heads, max_seq_len * 2, rope_theta
            ).to(device),
            requires_grad=False
        )

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int, seqlen: int):
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
        )
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], 2, -1).transpose(-2, -1).contiguous()
        )
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_).to(xq_.device)
        
        xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-2, -1).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-2, -1).flatten(3)
        
        return xq_out.type_as(xq), xk_out.type_as(xk)

class ALiBi(nn.Module):
    def __init__(self, n_heads, max_seq_len, device, alibi_bias_max=8):
        super(ALiBi, self).__init__()
        
        # Initialize ALiBi slopes and bias
        slopes, bias = self.build_alibi_bias(n_heads, max_seq_len, alibi_bias_max=alibi_bias_max)
        self.slopes = nn.Parameter(slopes.float().to(device), requires_grad=False)
        self.bias = nn.Parameter(bias.float().to(device), requires_grad=False)

    @staticmethod
    def gen_slopes(n_heads, alibi_bias_max=8):
        _n_heads = 2 ** math.ceil(math.log2(n_heads))
        m = torch.arange(1, _n_heads + 1, dtype=torch.float32)
        m = m.mul(alibi_bias_max / _n_heads)
        slopes = 1.0 / torch.pow(2, m)
        
        if _n_heads != n_heads:
            slopes = torch.cat([slopes[1::2], slopes[::2]])[:n_heads]
            
        return slopes.view(1, n_heads, 1, 1)

    @staticmethod
    def build_alibi_bias(n_heads, seq_len, alibi_bias_max=8, dtype=torch.float32):
        alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32).view(1, 1, 1, seq_len)
        slopes = ALiBi.gen_slopes(n_heads, alibi_bias_max)
        alibi_bias = alibi_bias * slopes
        slopes = slopes.squeeze(0).squeeze(-1).squeeze(-1)
        return slopes.to(dtype=dtype), alibi_bias.to(dtype=dtype)
    
    def forward(self, scores, seqlen):
        scores += self.bias[..., :seqlen]
        return scores

class QuantAttentionFused(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads, qkv_layer, o_proj, dev, max_seq_len, 
                       use_alibi=False, attention_shapes=None, rope_theta=10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_kv_groups = n_heads // n_kv_heads if n_kv_heads != 0 else 0
        self.head_dim = self.hidden_size // n_heads
        self.qkv_proj = qkv_layer
        self.o_proj = o_proj
        self.start_pos = 0
        self.use_alibi = use_alibi
        self.cache_batch_size = int(os.getenv("AWQ_BATCH_SIZE", "1"))
        self.max_seq_len = max_seq_len
        self.is_hf_transformers = False
        self.rope_theta = rope_theta

        # attention shapes for self attention
        self.attention_shapes = get_attention_shapes(
            attention_shapes, max_seq_len, self.cache_batch_size, n_heads, n_kv_heads, self.head_dim
        )
        # cache store that rolls cache
        self.cache = WindowedCache(
            self.attention_shapes["cache_v"], self.attention_shapes["cache_k"], self.max_seq_len, dev
        )

        if use_alibi:
            self.alibi = ALiBi(n_heads, max_seq_len, dev)
            self.rotary_dim = 0
            self.is_neox = False
        else:
            self.alibi = None
            self.rope = RoPE(hidden_size, n_heads, max_seq_len, dev, rope_theta)
            self.rotary_dim = self.head_dim
            self.is_neox = True
    
    def forward(self, hidden_states:torch.Tensor, attention_mask=None, *args, **kwargs):
        bsz, seqlen, _ = hidden_states.shape

        # Reallocate cache if batch size changes
        if bsz != self.cache_batch_size:
            if bsz > self.cache_batch_size:
                self.cache.increase_batch_size(bsz)
                self.cache_batch_size = bsz
            elif bsz < self.cache_batch_size:
                self.cache.decrease_batch_size(bsz)
                self.cache_batch_size = bsz

            # Always reset to 0
            self.start_pos = 0 

        # In case we re-generate, we need to refresh the starting position 
        # to 0. We detect it by checking if `past_key_values` is set to None, 
        # which indicates that we are on the first step of `generate()`.
        # This is only applicable for `transformers` integration
        if self.is_hf_transformers and "past_key_value" in kwargs and kwargs["past_key_value"] is None:
            self.start_pos = 0

        xqkv = self.qkv_proj(hidden_states)
        xqkv = xqkv.view((bsz, seqlen) + self.attention_shapes["xqkv_view"])
        
        xq = self.attention_shapes["xq_slice"](xqkv)
        xk = self.attention_shapes["xk_slice"](xqkv)
        xv = self.attention_shapes["xv_slice"](xqkv)

        if seqlen > 1 or not FT_INSTALLED:
            xq = xq.view((bsz, seqlen) + self.attention_shapes["xq_view"])
            xk = xk.view((bsz, seqlen) + self.attention_shapes["xk_view"])
            xv = xv.view((bsz, seqlen) + self.attention_shapes["xv_view"])

            if not self.use_alibi:
                xq, xk = self.rope.forward(xq, xk, self.start_pos, seqlen)

            self.cache.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape((bsz, seqlen) + self.attention_shapes["xk_reshape"])
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )
            
            self.cache.update_kv(values_store, keys_store, bsz, self.start_pos, seqlen)

            # Only necessary to retrieve from cache when we are not processing context
            if seqlen == 1:
                xv, xk = self.cache.get_kv(bsz, self.start_pos, seqlen, self.head_dim)

            
            keys = xk
            values = xv

            if self.n_kv_groups != 0:
                keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_kv_groups)
                values = torch.repeat_interleave(values, dim=2, repeats=self.n_kv_groups)
            
            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

            if self.use_alibi:
                scores = self.alibi.forward(scores, seqlen)

            # When seqlen is 1, there is nothing else to attend to
            if attention_mask is not None and seqlen > 1:
                scores = scores + attention_mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            attention_weight = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
            xq = xq.view((bsz,) + self.attention_shapes["single_xq_view"])
            xk = xk.view((bsz,) + self.attention_shapes["single_xk_view"])
            xv = xv.view((bsz,) + self.attention_shapes["single_xv_view"])

            alibi_slopes = self.alibi.slopes if self.alibi is not None else None
            attention_weight = awq_ft_ext.single_query_attention(
                xq, # query
                xk, # key
                xv, # value
                self.cache.k, # key cache
                self.cache.v, # value cache
                None, # length per sample
                alibi_slopes, # alibi slopes
                self.start_pos, # timestep
                self.rotary_dim, # rotary embedding dimension
                self.rope_theta, # rotary embedding base
                self.is_neox, # is neox
            )
            attention_weight = attention_weight.reshape(bsz, 1, -1)
        
        attn_output = self.o_proj(attention_weight)
        self.start_pos += seqlen

        # past_key_value is replaced with cache_v, cache_k, returning empty data
        # we pass a dummy past kv cache for transformers to be able to retrieve the correct info 
        # about past key length
        past_key_value = [torch.zeros(1, 1, self.start_pos, 1)]

        if HF_NEW_CACHE_FORMAT and self.is_hf_transformers:
            new_cache = DynamicCache()
            new_cache.update(past_key_value[0], past_key_value[0], layer_idx=0)
            past_key_value = new_cache

        return attn_output, attention_weight, past_key_value
