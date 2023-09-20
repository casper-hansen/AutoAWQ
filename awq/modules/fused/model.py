import torch
import torch.nn as nn
from awq.modules.fused.block import MPTBlock, FalconDecoderLayer, GptBigCodeBlock
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions

class MPTModel(nn.Module):
    def __init__(self, vocab_size, blocks, wte, norm_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = wte
        self.blocks: list[MPTBlock] = nn.ModuleList(blocks)
        self.norm_f = norm_f
        self.attn_uses_sequence_id = False
        self.prefix_lm = False

    @torch.inference_mode()
    def forward(self, input_ids, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        _bsz, seqlen = input_ids.shape
        h = self.wte(input_ids)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device
            )
            mask = torch.triu(mask, diagonal=self.blocks[0].attn.start_pos + 1).type_as(h)

        for layer in self.blocks:
            h, _, past_key_value = layer(h, None, attention_mask=mask, is_causal=is_causal)
        h = self.norm_f(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())

class FalconModel(nn.Module):
    def __init__(self, vocab_size, blocks, word_embeddings, ln_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embeddings = word_embeddings
        self.blocks: list[FalconDecoderLayer] = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.attn_uses_sequence_id = False
        self.prefix_lm = False

    @torch.inference_mode()
    def forward(self, input_ids, attn_bias=None, attention_mask=None, is_causal=None, *args, **kwargs):
        # NOTE: falcon input ids contain full context
        # after context is processed, slice to latest token
        if self.blocks[0].attn.start_pos != 0 and input_ids.shape[-1] != 1:
            input_ids = input_ids[:, self.blocks[0].attn.start_pos:]

        _bsz, seqlen = input_ids.shape
        h = self.word_embeddings(input_ids)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device
            )
            mask = torch.triu(mask, diagonal=self.blocks[0].attn.start_pos + 1).type_as(h)

        for layer in self.blocks:
            h, _, past_key_value = layer(h, None, attention_mask=mask, is_causal=is_causal)
        h = self.ln_f(h)

        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=())

class GptBigCodeModel(nn.Module):
    def __init__(self, vocab_size, blocks, wte, wpe, ln_f):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = wte
        self.wpe = wpe
        self.blocks: list[GptBigCodeBlock] = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.attn_uses_sequence_id = False
        self.prefix_lm = False

    @torch.inference_mode()
    def forward(self, input_ids, position_ids=None, token_type_ids=None, attention_mask=None, past_key_values=None, is_causal=None, *args, **kwargs):
        _bsz, seqlen = input_ids.shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.blocks))
        else:
            past_length = past_key_values[0].size(-2)
        
        if attention_mask is not None and len(attention_mask.shape) == 2 and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[:, past_length : input_ids.size()[-1] + past_length :]
        elif position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size()[-1] + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_ids.size()[-1])

        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        h = input_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            h = h + token_type_embeds

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=input_ids.device
            )
            mask = torch.triu(mask, diagonal=self.blocks[0].attn.start_pos + 1).type_as(h)

        for layer in self.blocks:
            h, _, past_key_value = layer(h, None, attention_mask=mask, is_causal=is_causal)
        h = self.ln_f(h)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=h, past_key_values=past_key_value, hidden_states=(), attentions=(), cross_attentions=()
        )
