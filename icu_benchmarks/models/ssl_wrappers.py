import torch
import torch.nn as nn
import gin

from icu_benchmarks.models.ssl.memory import MemoryQueue
from icu_benchmarks.models.ssl.loss import NCLLoss
from icu_benchmarks.models.wrappers import BaseModule
from icu_benchmarks.constants import RunMode

@gin.configurable("NCLWrapper")
class NCLWrapper(BaseModule):
    _supported_run_modes = [RunMode.self_supervised]

    def __init__(self,
                 encoder: nn.Module,
                 projector: nn.Module,
                 loss_fn: nn.Module = NCLLoss(),
                 projection_dim: int = 64,
                 queue_size: int = 65536,
                 run_mode=RunMode.self_supervised):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.loss_fn = loss_fn
        self.projection_dim = projection_dim
        self.queue_size = queue_size
        self.run_mode = run_mode

        self.memory_queue = None  # will be created in setup()

    def setup(self, stage=None):
        # Initialize memory queue on correct device
        self.memory_queue = MemoryQueue(
            embedding_dim=self.projection_dim,
            queue_size=self.queue_size,
            device=self.device
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.projector(z)
        z = nn.functional.normalize(z, dim=-1)
        return z

    def step_fn(self, batch, step_prefix="train"):
        x1, x2, key = batch  # x1, x2: (B, L, D); key: (B, 2)

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        queue, queue_keys = self.memory_queue.get()
        loss, agg_loss, disc_loss, acc = self.loss_fn(z1, queue, key, queue_keys)

        # Log
        self.log(f"{step_prefix}/loss", loss)
        self.log(f"{step_prefix}/agg_loss", agg_loss)
        self.log(f"{step_prefix}/disc_loss", disc_loss)
        self.log(f"{step_prefix}/accuracy", acc)

        # Update memory with z2
        self.memory_queue.enqueue(z2.detach(), key.detach())

        return loss
