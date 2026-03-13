from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch.distributed as dist
import typer
import yaml
from torch.utils.data.distributed import DistributedSampler

from scaletrain.data.datamodule import MNISTDataConfig, MNISTDataModule
from scaletrain.models.cnn import MNISTCNN
from scaletrain.tracking.mlflow_logger import MLflowConfig, MLflowLogger
from scaletrain.training.trainer import Trainer, TrainingConfig

log = logging.getLogger(__name__)

app = typer.Typer()

_BENCHMARK_EPOCHS = 2


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def __init__(self, rank: int = 0) -> None:
        super().__init__()
        self._rank = rank

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "rank": self._rank,
                "message": record.getMessage(),
            }
        )


def _setup_logging(rank: int) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter(rank=rank))

    root = logging.getLogger("scaletrain")
    root.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    # src/scaletrain/training/train.py -> src/scaletrain/configs/train_config.yaml
    return Path(__file__).resolve().parents[1] / "configs" / "train_config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter("Config YAML must be a mapping at the top level.")
    return data


def _section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = cfg.get(key, {}) or {}
    if not isinstance(value, dict):
        raise typer.BadParameter(f"Config section '{key}' must be a mapping.")
    return value


def _init_distributed() -> tuple[int, int]:
    """Initialize the default process group. Returns (rank, world_size)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")
    return rank, world_size


def _teardown_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _print_benchmark_summary(trainer: Trainer, world_size: int) -> None:
    mode = f"DDP ({world_size} proc)" if world_size > 1 else "single-process"
    sep = "─" * 46
    print("", flush=True)
    print(sep, flush=True)
    print("  Benchmark Results", flush=True)
    print(sep, flush=True)
    print(f"  Mode            : {mode}", flush=True)
    print(f"  Epochs          : {trainer.cfg.epochs}", flush=True)
    print(f"  Total time      : {trainer.total_training_time:.2f}s", flush=True)
    print(f"  Avg samples/s   : {trainer.avg_throughput:,.0f}", flush=True)
    print(sep, flush=True)
    print("", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    config: Path = typer.Option(
        _default_config_path(),
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to training YAML config.",
    ),
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        is_flag=True,
        help="Run in benchmark mode: 2 epochs, no MLflow logging, prints throughput summary.",
    ),
) -> None:
    cfg = _load_yaml(config)

    data_cfg = MNISTDataConfig(**_section(cfg, "data"))
    train_cfg = TrainingConfig(**_section(cfg, "training"))
    mlflow_cfg = MLflowConfig(**_section(cfg, "mlflow"))

    # --benchmark overrides config; forces a short fixed run with no side effects.
    if benchmark:
        train_cfg = dataclasses.replace(train_cfg, benchmark=True, epochs=_BENCHMARK_EPOCHS)

    rank = 0
    world_size = 1

    if train_cfg.distributed:
        rank, world_size = _init_distributed()

    _setup_logging(rank)
    log.info("training started  benchmark=%s", train_cfg.benchmark)

    dm = MNISTDataModule(data_cfg)

    # In distributed mode only rank 0 downloads; barrier ensures all ranks wait.
    if rank == 0:
        dm.prepare_data()
    if train_cfg.distributed:
        dist.barrier()
    dm.setup()

    model = MNISTCNN()

    sampler: Optional[DistributedSampler] = None
    if train_cfg.distributed:
        sampler = DistributedSampler(
            dm.train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    # Benchmark mode skips MLflow to measure raw training throughput.
    logger: Optional[MLflowLogger] = None
    tracking_cfg = _section(cfg, "tracking")
    if rank == 0 and bool(tracking_cfg.get("enabled", True)) and not train_cfg.benchmark:
        logger = MLflowLogger(mlflow_cfg)

    try:
        if logger:
            logger.start(
                params={
                    "data.batch_size": data_cfg.batch_size,
                    "data.num_workers": data_cfg.num_workers,
                    "training.epochs": train_cfg.epochs,
                    "training.lr": train_cfg.lr,
                    "training.weight_decay": train_cfg.weight_decay,
                    "training.device": train_cfg.device,
                    "training.distributed": train_cfg.distributed,
                    "training.world_size": world_size,
                }
            )

        trainer = Trainer(
            model=model,
            train_loader=dm.train_dataloader(sampler=sampler),
            val_loader=dm.val_dataloader(),
            cfg=train_cfg,
            logger=logger,
            rank=rank,
        )
        trainer.fit()

        if train_cfg.benchmark and rank == 0:
            _print_benchmark_summary(trainer, world_size)

        # Log the unwrapped model; `model` retains the trained weights regardless of DDP wrapping.
        if logger:
            logger.log_model(model, artifact_path="model")
    finally:
        if logger:
            logger.end()
        if train_cfg.distributed:
            _teardown_distributed()


if __name__ == "__main__":
    app()
