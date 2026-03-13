from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import mlflow


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: Optional[str] = None
    experiment_name: str = "scaletrain"
    run_name: Optional[str] = None


class MLflowLogger:
    """
    Thin wrapper around MLflow to keep training code clean.
    """

    def __init__(self, cfg: MLflowConfig) -> None:
        self.cfg = cfg
        self._active = False

    def start(self, params: Optional[Mapping[str, Any]] = None) -> None:
        if self.cfg.tracking_uri:
            mlflow.set_tracking_uri(self.cfg.tracking_uri)
        mlflow.set_experiment(self.cfg.experiment_name)
        mlflow.start_run(run_name=self.cfg.run_name)
        self._active = True
        if params:
            mlflow.log_params({k: self._stringify(v) for k, v in params.items()})

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if not self._active:
            return
        mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        if not self._active:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if not self._active:
            return
        # Uses MLflow's PyTorch integration (available when torch is installed).
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)

    def end(self) -> None:
        if self._active:
            mlflow.end_run()
        self._active = False

    @staticmethod
    def _stringify(v: Any) -> str:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return str(v)
        return repr(v)

