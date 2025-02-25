import torch


class WindowedCache:
    def __init__(
        self, cache_batch_size, n_heads, n_kv_heads, head_dim, max_seq_len, device
    ):
        """
        The window size is the same as the max_seq_len. The window will
        automatically roll once max_seq_len is exceeded.
        """
        size = (
            cache_batch_size,
            max_seq_len,
            n_kv_heads if n_kv_heads != 0 else n_heads,
            head_dim,
        )
        self.v = torch.zeros(
            size,
            device=device,
            dtype=torch.float16,
        )
        self.k = torch.zeros(
            size,
            device=device,
            dtype=torch.float16,
        )
        self.max_seq_len = max_seq_len

    def get_kv(self, batch_size, start_pos, seqlen):
        """
        Gets the key-value store in correct shapes.
        NOTE: This function is a legacy function. It is only available to showcase
              how to accurately retrieve the KV-cache but is not currently used.
        """
        xv = self.v[:batch_size, : start_pos + seqlen]
        xk = self.k[:batch_size, : start_pos + seqlen]

        return xv, xk

    def update_kv(self, values_store, keys_store, batch_size, start_pos, seqlen):
        """
        Updates the values in the key-value store.
        """
        self.v[:batch_size, start_pos : start_pos + seqlen, :, :] = values_store
        self.k[:batch_size, start_pos : start_pos + seqlen, :, :] = keys_store

    def roll_kv_n_steps(self, start_pos, n=100):
        """
        Roll cache n to the left.
        """
        n = min(n, self.max_seq_len)
        # Roll cache to the left
        self.v = torch.roll(self.v, shifts=-n, dims=2)
        self.k = torch.roll(self.k, shifts=-n, dims=2)

        # Zero out the new part
        self.v[:, :, -n:, :] = 0
        self.k[:, :, -n:, :] = 0

        return start_pos - n

    def to(self, device):
        self.k = self.k.to(device)
        self.v = self.v.to(device)

    def increase_batch_size(self, to_bsz):
        """Dynamically allocate new kv when batch size changes."""
        self.v = torch.zeros(
            to_bsz, *self.v.shape[1:], dtype=self.v.dtype, device=self.v.device
        )
        self.k = torch.zeros(
            to_bsz, *self.k.shape[1:], dtype=self.k.dtype, device=self.k.device
        )

    def decrease_batch_size(self, to_bsz):
        """Dynamically remove part of cache if batch size changes."""
        self.v = self.v[:to_bsz, :, :, :]
        self.k = self.k[:to_bsz, :, :, :]
