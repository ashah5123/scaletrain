from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import typer
import yaml

from scaletrain.data.datamodule import MNISTDataConfig, MNISTDataModule
from scaletrain.models.cnn import MNISTCNN
from scaletrain.tracking.mlflow_logger import MLflowConfig, MLflowLogger
from scaletrain.training.trainer import Trainer, TrainingConfig


app = typer.Typer(
    add_completion=False,
    help="ScaleTrain training CLI (single-process).",
    no_args_is_help=False,
)


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


@app.callback(invoke_without_command=True)
def run(
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
) -> None:
    print("=== CLI STARTED ===")
    print("=== LOADING CONFIG ===")

    cfg = _load_yaml(config)

    data_cfg = MNISTDataConfig(**_section(cfg, "data"))
    train_cfg = TrainingConfig(**_section(cfg, "training"))
    mlflow_cfg = MLflowConfig(**_section(cfg, "mlflow"))

    dm = MNISTDataModule(data_cfg)
    print("=== PREPARING DATA ===")
    dm.prepare_data()
    print("=== SETTING UP DATA ===")
    dm.setup()

    model = MNISTCNN()
    print("=== MODEL INITIALIZED ===")

    logger: Optional[MLflowLogger] = None
    tracking_cfg = _section(cfg, "tracking")
    if bool(tracking_cfg.get("enabled", True)):
        logger = MLflowLogger(mlflow_cfg)
    print("=== LOGGER INITIALIZED ===")

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
                }
            )

        trainer = Trainer(
            model=model,
            train_loader=dm.train_dataloader(),
            val_loader=dm.val_dataloader(),
            cfg=train_cfg,
            logger=logger,
        )
        print("=== STARTING TRAINING ===")
        trainer.fit()
        print("=== TRAINING FINISHED ===")

        if logger:
            logger.log_model(model, artifact_path="model")
    finally:
        if logger:
            logger.end()


if __name__ == "__main__":
    app()