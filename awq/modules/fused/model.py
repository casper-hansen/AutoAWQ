import torch
import torch.nn as nn
from typing import List
from awq.utils import fused_utils
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from awq.modules.fused.block import MPTBlock, FalconDecoderLayer, LlamaLikeBlock, MixtralBlock


class MixtralModel(nn.Module):
    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[MixtralBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attn_bias=None,
        attention_mask=None,
        is_causal=None,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h,
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, past_key_value = layer(h, None, attention_mask=mask, is_causal=is_causal)
        
        h = self.norm(h)

        return MoeModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=past_key_value,
            hidden_states=(),
            attentions=(),
            router_logits=(),
        )


class LlamaLikeModel(nn.Module):
    """
    LlamaLikeModel is intended to be reused across models that have
    an architecture that closely resembles Llama, e.g. Mistral and Aquila.
    """
    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[LlamaLikeBlock] = blocks
        self.norm = norm
        self.last_forward_num_tokens = 0
    
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids,
            self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, past_key_value = layer(
                h,
                None,
                attention_mask=mask,
                is_causal=is_causal
            )
        h = self.norm(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())

class MPTModel(nn.Module):
    def __init__(self, vocab_size, blocks, wte, norm_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = wte
        self.blocks: List[MPTBlock] = nn.ModuleList(blocks)
        self.norm_f = norm_f
        self.attn_uses_sequence_id = False
        self.prefix_lm = False
        self.last_forward_num_tokens = 0

    @torch.inference_mode()
    def forward(self, input_ids, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids,
            self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.wte(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, past_key_value = layer(
                h,
                None,
                attention_mask=mask,
                is_causal=is_causal
            )
        h = self.norm_f(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())

class FalconModel(nn.Module):
    def __init__(self, vocab_size, blocks, word_embeddings, ln_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embeddings = word_embeddings
        self.blocks: List[FalconDecoderLayer] = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.attn_uses_sequence_id = False
        self.prefix_lm = False
        self.last_forward_num_tokens = 0

    @torch.inference_mode()
    def forward(self, input_ids, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids,
            self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.word_embeddings(input_ids)

        mask = fused_utils.prepare_attention_mask(
            seqlen=seqlen,
            start_pos=self.blocks[0].attn.start_pos,
            device=input_ids.device,
            type_as=h
        )

        for layer in self.blocks:
            h, mask = fused_utils.prepare_correct_devices(
                layer,
                h,
                mask,
            )
            h, _, past_key_value = layer(
                h, 
                None, 
                attention_mask=mask, 
                is_causal=is_causal
            )
        h = self.ln_f(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())
