import torch
from typing import Dict

try:
    import awq_ext  # with CUDA kernels

    AWQ_INSTALLED = True
except:
    AWQ_INSTALLED = False


class FusedSparseMoeBlock(torch.nn.Module):
    def __init__(
        self,
        top_k,
        gate,
        ws,
        w2s,
    ):
        super().__init__()
        self.gate = gate
        self.top_k = top_k
        self.ws = ws
        self.w2s = w2s

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        final_hidden_states = apply_moe_weights(
            self.ws,
            self.w2s,
            hidden_states,
            router_logits,
            self.top_k,
            renormalize=True,
        )

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def apply_moe_weights(
    w1: Dict[str, torch.Tensor],
    w2: Dict[str, torch.Tensor],
    x: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(gating_output, topk, renormalize)
    (sorted_token_ids, expert_ids, num_tokens_post_padded) = moe_align_block_size(
        topk_ids, 16, w1.qweight.shape[0]
    )

    x = x.view(x.shape[0], 1, *x.shape[1:])

    gate_up = awq_ext.grouped_gemm_forward(
        x,
        w1.qweight,
        w1.scales,
        w1.qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        8,
    )

    out = torch.empty(
        (gate_up.shape[:-1] + (gate_up.shape[-1] // 2,)), dtype=x.dtype, device=x.device
    )
    awq_ext.silu_and_mul(out, gate_up)

    out = awq_ext.grouped_gemm_forward(
        out,
        w2.qweight,
        w2.scales,
        w2.qzeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        8,
    )

    return torch.sum(out, dim=1)



def moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    awq_ext.moe_alig_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """Compute top-k indice and weights from gating logits

    Args:
        gating_output (torch.Tensor): The output of the gating operation (before softmax).
        topk (int): The number of top-k experts to select.
        renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    """
    M = gating_output.shape[0]
    if torch.version.hip is not None:
        # The MoE kernels are not yet supported on ROCm.
        routing_weights = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    else:
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=gating_output.device
        )
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=gating_output.device)
        token_expert_indicies = torch.empty(
            M, topk, dtype=torch.int32, device=gating_output.device
        )
        awq_ext.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output.float(),  # TODO(woosuk): Optimize this.
        )
        del token_expert_indicies  # Not used. Will be used in the future.
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids
