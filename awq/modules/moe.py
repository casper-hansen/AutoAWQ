import torch
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

class ScaledMixtralSparseMoeBlock(torch.nn.Module):
    """
    This is a modified sparse MoE that scales experts individually.

    Modified version of:
    transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
    """

    def __init__(self, prev_op: MixtralSparseMoeBlock, scales: torch.Tensor):
        super().__init__()
        self.hidden_dim = prev_op.hidden_dim
        self.ffn_dim = prev_op.ffn_dim
        self.num_experts = prev_op.num_experts
        self.top_k = prev_op.top_k

        # gating
        self.gate = prev_op.gate

        # experts
        self.experts = prev_op.experts

        # [expert_num, hidden_dim]
        self.scales = torch.nn.Parameter(scales.data)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            
            ### NOTE: We scale weights here, modified from original MoE.
            current_state = hidden_states[None, top_x_list].reshape(
                -1, hidden_dim) / self.scales[expert_idx] # <-- scales

            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
