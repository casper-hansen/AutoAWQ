import os
import math
import torch
import torch.nn as nn
import awq_inference_engine
from torch.nn import functional as F

try:
    import ft_inference_engine
    FT_INSTALLED = True
except:
    FT_INSTALLED = False

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


class QuantAttentionFused(nn.Module):
    def __init__(self, hidden_size, n_heads, n_kv_heads, qkv_layer, o_proj, dev, max_seq_len, 
                       use_alibi=False, attention_shapes=None):
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

        if attention_shapes is not None:
            self.attention_shapes = attention_shapes

        elif self.n_kv_heads == 0:
            self.attention_shapes = {
                # following fastertransformer definition
                "cache_v": (self.cache_batch_size, self.n_heads, max_seq_len, self.head_dim,),
                # 8: pack 8 fp16 in FT, if fp32 then use 4
                "cache_k": (self.cache_batch_size, self.n_heads, self.head_dim // 8, max_seq_len, 8,),
                "xqkv_view": (-1, self.n_heads, self.head_dim),
                "xq_slice": lambda xqkv: xqkv[:, :, 0],
                "xk_slice": lambda xqkv: xqkv[:, :, 1],
                "xv_slice": lambda xqkv: xqkv[:, :, 2],
                "xq_view": (self.n_heads, self.head_dim),
                "xk_view": (self.n_heads, self.head_dim),
                "xv_view": (self.n_heads, self.head_dim),
                "xk_reshape": (self.n_heads, self.head_dim // 8, 8),
                "single_xq_view": (self.n_heads, self.head_dim),
                "single_xk_view": (self.n_heads, self.head_dim),
                "single_xv_view": (self.n_heads, self.head_dim)
            }

        else:
            self.attention_shapes = {
                # following fastertransformer definition
                "cache_v": (self.cache_batch_size, self.n_kv_heads, max_seq_len, self.head_dim,),
                # 8: pack 8 fp16 in FT, if fp32 then use 4
                "cache_k": (self.cache_batch_size, self.n_kv_heads, self.head_dim // 8, max_seq_len, 8,),
                "xqkv_view": (self.n_heads + self.n_kv_heads * 2, self.head_dim),
                "xq_slice": lambda xqkv: xqkv[:, :, 0 : self.n_heads],
                "xk_slice": lambda xqkv: xqkv[:, :, self.n_heads : (self.n_heads + self.n_kv_heads)],
                "xv_slice": lambda xqkv: xqkv[:, :, -self.n_kv_heads :],
                "xq_view": (self.n_heads, self.head_dim),
                "xk_view": (self.n_kv_heads, self.head_dim),
                "xv_view": (self.n_kv_heads, self.head_dim),
                "xk_reshape": (self.n_kv_heads, self.head_dim // 8, 8),
                "single_xq_view": (self.n_heads, self.head_dim),
                "single_xk_view": (self.n_kv_heads, self.head_dim),
                "single_xv_view": (self.n_kv_heads, self.head_dim)
            }

        self.cache_v = (
            torch.zeros(self.attention_shapes["cache_v"]).to(dev).half()
        )
        
        self.cache_k = (
            torch.zeros(self.attention_shapes["cache_k"]).to(dev).half()
        )

        if use_alibi:
            alibi_slopes, alibi_bias = build_alibi_bias(self.n_heads, max_seq_len)
            self.alibi_slopes = alibi_slopes.float().to(dev)
            self.alibi_bias = alibi_bias.float().to(dev)
            self.rotary_dim = 0
            self.is_neox = False
        else:
            self.freqs_cis = precompute_freqs_cis(
                hidden_size // n_heads,
                max_seq_len * 2,
            ).to(dev)
            self.rotary_dim = self.head_dim
            self.alibi_slopes = None
            self.is_neox = True
    
    def forward(
        self,
        hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False
    ):
        bsz, seqlen, _ = hidden_states.shape
        if bsz != self.cache_batch_size:
            raise RuntimeError(
                f"Batch size is incorrectly set - input batch size {bsz}, kv-cache batch size {self.cache_batch_size}. "
                f"Use: AutoAWQForCausalLM.from_quantized(batch_size={bsz})"
            )
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
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis[self.start_pos : self.start_pos + seqlen])

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            values_store = xv.transpose(2, 1)
            keys_store = (
                xk.reshape((bsz, seqlen) + self.attention_shapes["xk_reshape"])
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )

            self.cache_v[:bsz, :, self.start_pos : self.start_pos + seqlen, :] = values_store
            self.cache_k[:bsz, :, :, self.start_pos : self.start_pos + seqlen, :] = keys_store

            if seqlen == 1:
                xv = self.cache_v[:bsz, :, : self.start_pos + seqlen, :].transpose(1, 2).contiguous()
                xk = self.cache_k[:bsz, :, :, : self.start_pos + seqlen, :].transpose(2, 3).contiguous()
                xk = xk.reshape(xk.shape[:-2] + (self.head_dim,)).transpose(1, 2).contiguous()
            
            keys = xk
            values = xv

            if self.n_kv_groups != 0:
                keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_kv_groups)
                values = torch.repeat_interleave(values, dim=2, repeats=self.n_kv_groups)
            
            past_key_value = (xk, xv) if use_cache else None
            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

            if self.use_alibi:
                scores += self.alibi_bias[..., :seqlen]

            if attention_mask is not None:
                scores = scores + attention_mask  # (bs, n_local_heads, slen, cache_len + slen)
                
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            attention_weight = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        else:
            # xq = xq[:, 0, :, :]
            # xk = xk[:, 0, :, :]
            # xv = xv[:, 0, :, :]
            xq = xq.view((bsz,) + self.attention_shapes["single_xq_view"])
            xk = xk.view((bsz,) + self.attention_shapes["single_xk_view"])
            xv = xv.view((bsz,) + self.attention_shapes["single_xv_view"])

            past_key_value = (xk, xv) if use_cache else None
            attention_weight = ft_inference_engine.single_query_attention(
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
            attention_weight = attention_weight.reshape(bsz, 1, -1)
        
        attn_output = self.o_proj(attention_weight)
        
        if use_cache:
            self.start_pos += seqlen
        else:
            self.start_pos = 0

        return attn_output, attention_weight, past_key_value
