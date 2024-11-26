import torch
import torch.nn as nn
from typing import List
from awq.utils import fused_utils
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
)
from awq.modules.fused.block import (
    MPTBlock,
    FalconDecoderLayer,
    LlamaLikeBlock,
    MixtralBlock,
    Phi3Block,
    CohereBlock,
    Gemma2LikeBlock,
)


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
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        for layer in self.blocks:
            h = h.to(layer.device)
            h = layer(h)

        h = self.norm(h)

        return MoeModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
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
        self.blocks: List[LlamaLikeBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @property
    def embed_tokens(self):
        return self.embedding

    @property
    def layers(self):
        return self.blocks

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        for layer in self.blocks:
            h = h.to(layer.device)
            h = layer(h)

        h = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )


class CohereModel(nn.Module):
    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[CohereBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @property
    def embed_tokens(self):
        return self.embedding

    @property
    def layers(self):
        return self.blocks

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

        for layer in self.blocks:
            h = h.to(layer.device)
            h = layer(h)

        h = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )


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
    def forward(
        self,
        input_ids,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.wte(input_ids)

        for layer in self.blocks:
            h = h.to(layer.device)
            h = layer(h)

        h = self.norm_f(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )


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
    def forward(
        self,
        input_ids,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.word_embeddings(input_ids)

        for layer in self.blocks:
            h = h.to(layer.device)
            h = layer(h)

        h = self.ln_f(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )


class Phi3Model(nn.Module):
    """
    Phi3LikeModel is intended to be reused across models that have
    an architecture that closely resembles Phi-3.
    """

    def __init__(self, vocab_size, blocks, embedding, norm):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[Phi3Block] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0

    @property
    def embed_tokens(self):
        return self.embedding

    @property
    def layers(self):
        return self.blocks

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        for layer in self.blocks:
            h = h.to(layer.device)
            h = layer(h)

        h = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )


class Gemma2LikeModel(nn.Module):
    def __init__(self, vocab_size, blocks, embedding, norm, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.blocks: List[Gemma2LikeBlock] = nn.ModuleList(blocks)
        self.norm = norm
        self.last_forward_num_tokens = 0
        self.hidden_size = hidden_size

    @property
    def embed_tokens(self):
        return self.embedding

    @property
    def layers(self):
        return self.blocks

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ):
        input_ids, self.last_forward_num_tokens = fused_utils.prepare_input_ids(
            input_ids, self.last_forward_num_tokens
        )
        _bsz, seqlen = input_ids.shape

        fused_utils.prepare_cache(self.blocks, seqlen)

        h = self.embedding(input_ids)

        normalizer = torch.tensor(self.hidden_size**0.5, dtype=h.dtype)
        h = h * normalizer

        for layer in self.blocks:
            h = h.to(layer.device)
            h = layer(h)

        h = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=h,
            past_key_values=None,
            hidden_states=(),
            attentions=(),
        )
