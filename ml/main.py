from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
import uvicorn

from predict_fixtures import ImprovedFixturePredictor


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = ImprovedFixturePredictor()

    print("🚀 Running predictions at startup...")
    predictions = await predictor.predict_all_fixtures()

    if predictions:
        predictor.display_predictions(predictions)

        # Save predictions to CSV
        df = pd.DataFrame(predictions)
        df.to_csv("improved_fixture_predictions.csv", index=False)
        print("💾 Predictions saved to improved_fixture_predictions.csv")
    else:
        print("❌ No predictions generated")

    # Yield control to the application
    yield

    # (Optional) Shutdown tasks can go here
    print("🛑 Server shutting down...")


# Initialize app with lifespan
app = FastAPI(lifespan=lifespan)


@app.get("/api/predictions")
async def get_predictions():
    try:
        predictor = ImprovedFixturePredictor()
        predictions = await predictor.predict_all_fixtures()

        return [
            {
                "fixture": p["fixture"],
                "predicted_outcome": p["predicted_outcome"],
                "confidence": p["outcome_confidence"],
                "over_under": p["over_25_prediction"],
                "over_under_confidence": p["over_25_confidence"],
                "insights": p.get("insights", []),
                "model_accuracy": p.get("model_accuracy", ""),
                "h2h_wins": p.get("h2h_wins", 0),
                "h2h_draws": p.get("h2h_draws", 0),
                "h2h_losses": p.get("h2h_losses", 0),
            }
            for p in predictions
        ]
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
