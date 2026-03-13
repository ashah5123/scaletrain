from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    inputs: List[List[float]]

    @field_validator("inputs")
    @classmethod
    def check_input_width(cls, v: List[List[float]]) -> List[List[float]]:
        for i, row in enumerate(v):
            if len(row) != 784:
                raise ValueError(
                    f"Each input must contain 784 values (28x28 image flattened); "
                    f"got {len(row)} values at index {i}."
                )
        return v


class PredictResponse(BaseModel):
    predictions: List[int]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _latest_run_uri(tracking_uri: str, experiment_name: str) -> str:
    """Return the artifact URI for the most recent finished run."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' not found. "
            "Run training at least once before starting the inference server."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No finished runs found in experiment '{experiment_name}'. "
            "Complete a training run before starting the inference server."
        )

    return f"runs:/{runs[0].info.run_id}/model"


def _resolve_model_uri(
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    version: Optional[str],
    stage: Optional[str],
) -> str:
    """
    Resolve the MLflow model URI according to loading priority:
      1. Explicit version  →  models:/<name>/<version>
      2. Stage alias       →  models:/<name>/<stage>
      3. Fallback          →  latest finished run artifact
    """
    if version is not None:
        return f"models:/{model_name}/{version}"
    if stage is not None:
        return f"models:/{model_name}/{stage}"
    return _latest_run_uri(tracking_uri, experiment_name)


def _load_model(tracking_uri: str, model_uri: str) -> torch.nn.Module:
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri  = os.environ.get("MLFLOW_TRACKING_URI",   "sqlite:///mlflow.db")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "scaletrain")
    model_name    = os.environ.get("MLFLOW_MODEL_NAME",      experiment_name)
    version       = os.environ.get("MODEL_VERSION")   # e.g. "3"
    stage         = os.environ.get("MODEL_STAGE")     # e.g. "Production"

    try:
        model_uri = _resolve_model_uri(tracking_uri, experiment_name, model_name, version, stage)
        print(f"[scaletrain] loading model: {model_uri}", flush=True)
        app.state.model = _load_model(tracking_uri, model_uri)
        print(f"[scaletrain] model ready:   {model_uri}", flush=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Startup failed: {exc}") from exc

    yield

    app.state.model = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="ScaleTrain Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    model: torch.nn.Module = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    tensor = torch.tensor(body.inputs, dtype=torch.float32).view(-1, 1, 28, 28)

    with torch.no_grad():
        logits = model(tensor)

    predictions: List[int] = logits.argmax(dim=1).tolist()
    return PredictResponse(predictions=predictions)
