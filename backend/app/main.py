from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .inference import (
    MODEL_METADATA, MODEL_ID, MAX_UPLOAD_BYTES,
    ImageValidationError, load_model, model_status, predict_bytes
)
from .schemas import BatchPredictionResponse, HealthResponse, PredictionResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup if packaged. Keep API bootable without weights for setup checks.
    if model_status():
        load_model()
    yield


app = FastAPI(
    title="Breast Cancer Histopathology AI API",
    version="0.1.0",
    description="Experimental research inference API. Not for clinical use.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Prediction"],  # /explain returns prediction JSON in this header
)


@app.get("/health", response_model=HealthResponse)
def health():
    ready = model_status()
    return HealthResponse(
        status="ready" if ready else "model_not_packaged",
        model_loaded=ready,
        model_id=MODEL_ID,
    )


@app.get("/model-info")
def model_info():
    return MODEL_METADATA


async def _read_upload(file: UploadFile) -> bytes:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 12 MB upload limit.")
    return raw


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    raw = await _read_upload(file)
    try:
        return predict_bytes(raw, file.filename or "upload")
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=422, detail="Upload at least one image.")
    if len(files) > 20:
        raise HTTPException(status_code=413, detail="A maximum of 20 images is allowed per batch.")

    results = []
    for file in files:
        raw = await _read_upload(file)
        try:
            results.append(predict_bytes(raw, file.filename or "upload"))
        except ImageValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"{file.filename or 'upload'}: {exc}"
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BatchPredictionResponse(count=len(results), results=results)


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    raw = await _read_upload(file)
    try:
        from .explainability import make_gradcam
        prediction, overlay = make_gradcam(raw)
        # Return prediction metadata in headers; overlay is the PNG response body.
        import json
        return Response(
            content=overlay,
            media_type="image/png",
            headers={"X-Prediction": json.dumps(prediction)},
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
