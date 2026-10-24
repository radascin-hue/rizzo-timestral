from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from timestral import TimestralForecaster

app = FastAPI(title="Rizzo Timestral")
model = TimestralForecaster.from_pretrained("mistralai/Timestral-5M")

class Req(BaseModel):
    history: list[float]
    horizon: int = 8

@app.get("/")
def root():
    return {"ok": True, "model": "Timestral-5M"}

@app.post("/forecast")
def forecast(req: Req):
    try:
        if len(req.history) < 12:
            return {"error": "servono almeno 12 punti storici"}
        ctx = np.asarray(req.history, dtype=np.float64)
        pred = model.forecast(ctx, prediction_length=req.horizon)
        med = np.asarray(pred.median).reshape(-1)
        q = np.asarray(pred.quantiles).reshape(-1, 21)
        return {
            "median": round(float(med[-1]), 3),
            "low": round(float(q[-1][2]), 3),
            "high": round(float(q[-1][18]), 3),
            "horizon": req.horizon,
        }
    except Exception as e:
        return {"error": str(e)}
