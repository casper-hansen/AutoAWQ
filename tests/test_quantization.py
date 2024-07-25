import torch
import numpy as np

def test_per_channel_mean(inp, max_chunk_memory=1024*1024*1024, atol=1e-5, rtol=1e-5):
    # Original method
    x_mean_original = inp.abs().view(-1, inp.shape[-1]).mean(0)

    # New method with chunking
    inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
    num_elements = inp_flat.size(0)
    num_channels = inp_flat.size(1)
    element_size_bytes = inp_flat.element_size() * 2

    chunk_size = int(max_chunk_memory // (element_size_bytes * num_channels))
    chunk_size = min(chunk_size, num_elements)

    x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
    
    for i in range(0, num_elements, chunk_size):
        end = min(i + chunk_size, num_elements)
        chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
        x_sum += chunk_sum.to(inp.device)

    x_mean_new = (x_sum / num_elements).to(inp.dtype)

    # Compare results
    are_close = torch.allclose(x_mean_original, x_mean_new, atol=atol, rtol=rtol)
    max_diff = torch.max(torch.abs(x_mean_original - x_mean_new)).item()

    print(f"Results are close: {are_close}")
    print(f"Maximum difference: {max_diff}")

    return are_close


def pseudo_quantize_tensor(w: torch.Tensor, group_size=128, w_bit=4):
    org_w_shape = w.shape
    if group_size > 0:
        assert org_w_shape[-1] % group_size == 0
        w = w.reshape(-1, group_size)
    assert w.dim() == 2
    assert torch.isnan(w).sum() == 0

    # zero point quantization
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**w_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales
    zeros = zeros.view(org_w_shape[0], -1)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    scales = scales.view(org_w_shape[0], -1)
    w = w.reshape(org_w_shape)

    return w, scales, zeros


def test_loss_computation(fp16_output, int_w_output, max_chunk_memory=1024*1024*1024, atol=1e-5, rtol=1e-5):
    # Original method
    loss_original = (fp16_output - int_w_output).float().pow(2).mean().item()

    # New method with chunking
    @torch.no_grad()
    def _compute_loss(fp16_output, int_w_output, device, max_chunk_memory):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        chunk_size = max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        loss /= num_elements
        return loss

    loss_new = _compute_loss(fp16_output, int_w_output, fp16_output.device, max_chunk_memory)

    # Compare results
    are_close = np.isclose(loss_original, loss_new, atol=atol, rtol=rtol)
    diff = abs(loss_original - loss_new)

    print(f"Results are close: {are_close}")
    print(f"Difference: {diff}")

    return are_close

fp16_output = torch.randn(1000, 1000, 512)
int_w_output = pseudo_quantize_tensor(fp16_output)[0]
test_result = test_loss_computation(fp16_output, int_w_output)

inp = torch.randn(1000, 1000, 512)
test_result = test_per_channel_mean(inp)