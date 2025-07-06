from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model_outcome = joblib.load("model_outcome.pkl")
model_over = joblib.load("model_over25.pkl")
feature_list = joblib.load("feature_list.pkl")

class MatchRequest(BaseModel):
    home_form_wins: float
    away_form_wins: float
    elo_diff: float

@app.post("/predict/outcome")
def predict_outcome(req: MatchRequest):
    X = np.array([[req.home_form_wins, req.away_form_wins, req.elo_diff]])
    pred = model_outcome.predict(X)[0]
    proba = model_outcome.predict_proba(X)[0].tolist()
    outcome_map = {0:"home_win",1:"draw",2:"away_win"}
    return {"prediction": outcome_map[pred], "confidence": max(proba), "probabilities": proba}

@app.post("/predict/over25")
def predict_over(req: MatchRequest):
    X = np.array([[req.home_form_wins, req.away_form_wins, req.elo_diff]])
    pred = model_over.predict(X)[0]
    proba = model_over.predict_proba(X)[0].tolist()
    return {"prediction": bool(pred), "confidence": max(proba), "probabilities": proba} 