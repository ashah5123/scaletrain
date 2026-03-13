from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from scaletrain.tracking.mlflow_logger import MLflowLogger

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    log_every_n_steps: int = 50
    distributed: bool = False
    benchmark: bool = False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingConfig,
        logger: Optional[MLflowLogger] = None,
        rank: int = 0,
    ) -> None:
        self.rank = rank
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger

        # gloo on macOS does not support MPS; distributed mode always uses CPU.
        self.device = torch.device("cpu") if cfg.distributed else self._select_device(cfg.device)
        self.model.to(self.device)

        if cfg.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # Populated by fit(); available to callers afterwards.
        self._epoch_times: List[float] = []
        self._throughputs: List[float] = []
        self.total_training_time: float = 0.0

    @property
    def avg_throughput(self) -> float:
        if not self._throughputs:
            return 0.0
        return sum(self._throughputs) / len(self._throughputs)

    def fit(self) -> None:
        global_step = 0
        t_run_start = time.perf_counter()

        for epoch in range(1, self.cfg.epochs + 1):
            t_epoch_start = time.perf_counter()
            train_loss, _train_acc, global_step, n_samples = self._train_epoch(epoch, global_step)
            val_loss, val_accuracy = self._eval_epoch()
            epoch_time = time.perf_counter() - t_epoch_start
            samples_per_second = n_samples / max(epoch_time, 1e-9)

            self._epoch_times.append(epoch_time)
            self._throughputs.append(samples_per_second)

            if self.rank == 0:
                log.info(
                    "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  "
                    "epoch_time=%.2fs  samples/s=%.0f",
                    epoch, self.cfg.epochs,
                    train_loss, val_loss, val_accuracy,
                    epoch_time, samples_per_second,
                )

            if self.rank == 0 and self.logger:
                self.logger.log_metric("train_loss", float(train_loss), step=epoch)
                self.logger.log_metric("val_loss", float(val_loss), step=epoch)
                self.logger.log_metric("val_accuracy", float(val_accuracy), step=epoch)
                self.logger.log_metric("epoch_time_seconds", epoch_time, step=epoch)
                self.logger.log_metric("samples_per_second", samples_per_second, step=epoch)

        self.total_training_time = time.perf_counter() - t_run_start
        if self.rank == 0:
            log.info("training complete  total_time=%.2fs", self.total_training_time)
            if self.logger:
                self.logger.log_metric("total_training_time", self.total_training_time)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        return self._eval_epoch()

    def _train_epoch(self, epoch: int, global_step: int) -> Tuple[float, float, int, int]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # DistributedSampler requires set_epoch for proper shuffling each epoch.
        sampler = self.train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        for batch_idx, (x, y) in enumerate(self.train_loader, start=1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            bs = x.size(0)
            total += bs
            total_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == y).sum().item()

            global_step += 1
            if self.rank == 0 and self.logger and (global_step % self.cfg.log_every_n_steps == 0):
                self.logger.log_metrics(
                    {"train/loss_step": float(loss.item())},
                    step=global_step,
                )

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc, global_step, total

    @torch.no_grad()
    def _eval_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)

            bs = x.size(0)
            total += bs
            total_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    @staticmethod
    def _select_device(requested: str) -> torch.device:
        requested = (requested or "auto").lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            return torch.device("cuda")
        if requested == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("MPS requested but not available.")
            return torch.device("mps")

        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
