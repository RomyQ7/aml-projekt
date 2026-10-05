from pathlib import Path
import logging
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field


project_dir = Path(__file__).resolve().parent
log_dir = project_dir / "logs"
log_dir.mkdir(exist_ok=True)

# Protokoll für Anfragen, Laufzeiten und Warnungen.
logger = logging.getLogger("aml_api")
logger.setLevel(logging.INFO)

handler = logging.FileHandler(
    log_dir / "api.log", encoding="utf-8"
)
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
logger.addHandler(handler)

# Gespeichertes Modell einmal beim Start laden.
bundle = joblib.load(project_dir / "models" / "aml_model.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = float(bundle["threshold"])
positive_column = list(model.classes_).index(1)

app = FastAPI(
    title="AML-Prototyp",
    description="Bewertung synthetischer Finanztransaktionen",
    version="1.0",
)


# Regeln für eingehende Daten.
class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )
    amount: float = Field(gt=0, allow_inf_nan=False)
    new_recipient: int = Field(ge=0, le=1, strict=True)
    country_risk: float = Field(
        ge=0, le=1, allow_inf_nan=False
    )


class Prediction(BaseModel):
    transaction_id: str
    model_score: float
    threshold: float
    alert: bool


# Dauer und HTTP-Status jeder Anfrage protokollieren.
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        logger.exception("Fehler bei der Verarbeitung")
        raise
    finally:
        duration_ms = (perf_counter() - start) * 1000
        logger.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            status,
            duration_ms,
        )


# Einfache Erreichbarkeitsprüfung
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


# Eine einzelne Transaktion bewerten
@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    inputs = pd.DataFrame(
        [transaction.model_dump()],
        columns=features,
    )

    score = float(
        model.predict_proba(inputs)[0, positive_column]
    )
    alert = score >= threshold

    logger.info(
        "transaction_id=%s model_score=%.6f alert=%s",
        transaction.transaction_id,
        score,
        alert,
    )

    return Prediction(
        transaction_id=transaction.transaction_id,
        model_score=score,
        threshold=threshold,
        alert=alert,
    )