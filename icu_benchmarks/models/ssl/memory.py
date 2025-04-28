import torch

class MemoryQueue:
    def __init__(self, embedding_dim, queue_size=65536, device='cpu'):
        """
        Implements a FIFO memory queue for NCL self-supervised learning.

        Args:
            embedding_dim (int): Dimensionality of the embedding vectors.
            queue_size (int): Maximum number of elements to store.
            device (str): Device to store the queue (e.g., 'cuda' or 'cpu').
        """
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.device = device

        # Empty tensors to start with
        self.queue = torch.empty((0, embedding_dim), dtype=torch.float32, device=device)
        self.keys = torch.empty((0, 2), dtype=torch.int32, device=device)

    def enqueue(self, z_batch, key_batch):
        """
        Add a batch of projections and their corresponding keys.

        Args:
            z_batch (Tensor): (B, D) — projections
            key_batch (Tensor): (B, 2) — identity keys (stay_id, time)
        """
        self.queue = torch.cat([z_batch, self.queue], dim=0)[:self.queue_size]
        self.keys = torch.cat([key_batch, self.keys], dim=0)[:self.queue_size]

    def get(self):
        """
        Returns:
            queue (Tensor): (K, D)
            keys (Tensor): (K, 2)
        """
        return self.queue, self.keys

    def reset(self):
        """Reset the queue."""
        self.queue = torch.empty((0, self.embedding_dim), dtype=torch.float32, device=self.device)
        self.keys = torch.empty((0, 2), dtype=torch.int32, device=self.device)
