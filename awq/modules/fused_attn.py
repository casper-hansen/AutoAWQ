import torch
import torch.nn as nn
import awq_inference_engine
from torch.nn import functional as F
from torch.backends.cuda import sdp_kernel, SDPBackend

class QuantLlamaRotary(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, is_neox=True,
                       num_heads=None, head_dim=None, num_kv_heads=None):
        super().__init__()
        self.dim = dim
        self.is_neox = is_neox
        self.num_heads = num_heads
        self.head_dim = head_dim
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
    
    def forward(self, qkv_states: torch.Tensor, position_ids: torch.Tensor):
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

        # contiguous memory
        query = query.contiguous()
        key = key.contiguous()

        # apply vLLM kernel
        awq_inference_engine.rotary_embedding(
            positions,
            query,
            key,
            self.dim,
            self.cos_sin_cache,
            self.is_neox
        )

        # reshape output for attention
        query = query.view(query_batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(query_batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(query_batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)

        return query, key, value

class TorchAttention(nn.Module):
    def __init__(self, attention_type:int, hidden_size:int) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.backend_map = {
            SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
            SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
            SDPBackend.EFFICIENT_ATTENTION: {"enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
        }
        self.attn_config = self.backend_map[attention_type]
    
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, use_cache:bool, 
                      past_key_value:torch.Tensor, hidden_states_shape:torch.Size):
        batch_size, q_len, _ = hidden_states_shape
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

        with sdp_kernel(**self.attn_config):
            attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        
        del query, key, value

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q_len, self.hidden_size)

        return attn_output, None, past_key_value

class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size:int,
        num_heads:int,
        num_kv_heads:int,
        qkv_proj: torch.Tensor,
        o_proj: torch.Tensor,
        dev:str,
        max_new_tokens:int,
        attention_type=SDPBackend.FLASH_ATTENTION
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.attn = TorchAttention(attention_type, hidden_size)

        self.rotary_emb = QuantLlamaRotary(
            self.head_dim, 
            max_position_embeddings=max_new_tokens, 
            device=dev,
            is_neox=True,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads
        )

    def forward(self, hidden_states, past_key_value=None, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False):
        qkv_states = self.qkv_proj(hidden_states)
        query, key, value = self.rotary_emb(qkv_states, position_ids)
        attn_output, _, past_key_value = self.attn(query, key, value, use_cache, past_key_value, hidden_states.shape)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
