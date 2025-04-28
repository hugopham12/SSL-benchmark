import torch
import torch.nn.functional as F

class NCLLoss(torch.nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5, threshold=8):
        """
        PyTorch implementation of the Neighborhood Contrastive Loss (NCL).
        
        Args:
            temperature (float): Temperature scaling.
            alpha (float): Weight between aggregation and discrimination terms.
            threshold (int): Max temporal distance to define neighbors.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, q, queue, labels_q, labels_queue):
        """
        Args:
            q (Tensor): shape (B, D), current batch of projections
            queue (Tensor): shape (K, D), memory queue of projections
            labels_q (Tensor): shape (B, 2), batch identity keys (patient_id, time)
            labels_queue (Tensor): shape (K, 2), queue identity keys

        Returns:
            loss (Tensor): full NCL loss
            aggregation_loss (Tensor)
            discrimination_loss (Tensor)
            accuracy (Tensor)
        """
        B, D = q.shape
        K = queue.shape[0]

        # Compute logits (similarity between q and all in queue)
        logits = torch.matmul(q, queue.T) / self.temperature  # (B, K)

        # Compute direct positives (diagonal)
        k = queue[:B]  # positives are first B items
        pos = torch.bmm(q.view(B, 1, D), k.view(B, D, 1)).squeeze() / self.temperature  # (B,)

        # Define neighbors: same patient, within ±threshold hours
        pid_q, t_q = labels_q[:, 0], labels_q[:, 1]
        pid_k, t_k = labels_queue[:, 0], labels_queue[:, 1]

        # Expand for pairwise comparison
        pid_q = pid_q.view(B, 1)
        t_q = t_q.view(B, 1)
        pid_k = pid_k.view(1, K)
        t_k = t_k.view(1, K)

        same_patient = (pid_q == pid_k)
        close_in_time = (torch.abs(t_q - t_k) <= self.threshold)
        neighbors_mask = (same_patient & close_in_time).float()  # (B, K)

        # Avoid division by zero
        neighbor_count = neighbors_mask.sum(dim=1).clamp(min=1.0)

        # Aggregation term (logits vs neighbors)
        expectation_marginal = torch.logsumexp(logits, dim=1)  # (B,)
        expectation_neighbors = (logits * neighbors_mask).sum(dim=1) / neighbor_count  # (B,)
        aggregation_loss = (expectation_marginal - expectation_neighbors).mean()

        # Discrimination term (neighbors vs positive)
        expectation_neighbors_log = torch.log((torch.exp(logits) * neighbors_mask).sum(dim=1).clamp(min=1e-6))
        discrimination_loss = (expectation_neighbors_log - pos).mean()

        # Total loss
        total_loss = self.alpha * aggregation_loss + (1.0 - self.alpha) * discrimination_loss

        # Contrastive accuracy (argmax over logits)
        preds = logits.argmax(dim=1)
        targets = torch.arange(B, device=logits.device)
        accuracy = (preds == targets).float().mean()

        return total_loss, aggregation_loss, discrimination_loss, accuracy
