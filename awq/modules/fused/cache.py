import torch

class WindowedCache:
    def __init__(self, cache_v_shape, cache_k_shape, max_seq_len, device):
        """
        The window size is the same as the max_new_tokens. The window will
        automatically roll once max_new_tokens is exceeded.
        """
        # [batch_size, n_kv_heads, max_seq_len, head_dim]
        self.v = torch.zeros(cache_v_shape).to(device).half()
        # [batch_size, n_kv_heads, head_dim // pack_factor, max_seq_len, pack_factor]
        self.k = torch.zeros(cache_k_shape).to(device).half()
        self.max_seq_len = max_seq_len
    
    def get_kv(self, batch_size, start_pos, seqlen, head_dim):
        """
        Gets the key-value store in correct shapes.
        """
        xv = self.v[:batch_size, :, : start_pos + seqlen, :].transpose(1, 2).contiguous()
        xk = self.k[:batch_size, :, :, : start_pos + seqlen, :].transpose(2, 3).contiguous()
        xk = xk.reshape(xk.shape[:-2] + (head_dim,)).transpose(1, 2).contiguous()

        return xv, xk
    
    def update_kv(self, values_store, keys_store, batch_size, start_pos, seqlen):
        """
        Updates the values in the key-value store.
        """
        self.v[:batch_size, :, start_pos : start_pos + seqlen, :] = values_store
        self.k[:batch_size, :, :, start_pos : start_pos + seqlen, :] = keys_store

    def roll_kv_n_steps(self, start_pos, n=100):
        """
        Roll cache n to the left.
        """
        n = min(n, self.max_seq_len)
        # Roll cache to the left
        self.v = torch.roll(self.v, shifts=-n, dims=2)
        self.k = torch.roll(self.k, shifts=-n, dims=3)

        # Zero out the new part
        self.v[:, :, -n:, :] = 0
        self.k[:, :, :, -n:, :] = 0
        
        return start_pos - n
    
    def to(self, device):
        self.k = self.k.to(device)
        self.v = self.v.to(device)
    
    def increase_batch_size(self, to_bsz):
        """Dynamically allocate new kv when batch size changes."""
        self.v = torch.zeros(to_bsz, *self.v.shape[1:], dtype=self.v.dtype, device=self.v.device)
        self.k = torch.zeros(to_bsz, *self.k.shape[1:], dtype=self.k.dtype, device=self.k.device)

    def decrease_batch_size(self, to_bsz):
        """Dynamically remove part of cache if batch size changes."""
        self.v = self.v[:to_bsz, :, :, :]
        self.k = self.k[:to_bsz, :, :, :, :]