import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'recommendation-engine'))

from football_stats_agent import FootballStatsAgent
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FixturePredictor:
    def __init__(self):
        # Load the trained models
        self.model_outcome = joblib.load("model_outcome.pkl")
        self.model_over = joblib.load("model_over25.pkl")
        self.feature_list = joblib.load("feature_list.pkl")
        self.agent = FootballStatsAgent()
        
        # Outcome mapping
        self.outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    
    def extract_features(self, fixture_data: Dict) -> Dict:
        """Extract features from fixture data for ML prediction"""
        try:
            # Extract form data from team records
            home_record = fixture_data.get(f"{fixture_data['Fixture'].split(' vs ')[0]} Record", "0-0-0")
            away_record = fixture_data.get(f"{fixture_data['Fixture'].split(' vs ')[1]} Record", "0-0-0")
            
            # Parse W-D-L records
            home_w, home_d, home_l = map(int, home_record.split('-'))
            away_w, away_d, away_l = map(int, away_record.split('-'))
            
            # Calculate form (wins in last 5 matches)
            home_form_wins = home_w / max(1, home_w + home_d + home_l) * 3  # Normalize to 0-3 scale
            away_form_wins = away_w / max(1, away_w + away_d + away_l) * 3
            
            # Simple Elo-like difference based on goals scored
            home_goals = fixture_data.get(f"{fixture_data['Fixture'].split(' vs ')[0]} Goals", 0)
            away_goals = fixture_data.get(f"{fixture_data['Fixture'].split(' vs ')[1]} Goals", 0)
            elo_diff = home_goals - away_goals
            
            return {
                "home_form_wins": home_form_wins,
                "away_form_wins": away_form_wins,
                "elo_diff": elo_diff
            }
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {
                "home_form_wins": 1.5,
                "away_form_wins": 1.5,
                "elo_diff": 0
            }
    
    def predict_fixture(self, fixture_data: Dict) -> Dict:
        """Make predictions for a single fixture"""
        features = self.extract_features(fixture_data)
        
        # Prepare feature array
        X = np.array([[features["home_form_wins"], features["away_form_wins"], features["elo_diff"]]])
        
        # Get outcome prediction
        outcome_pred = self.model_outcome.predict(X)[0]
        outcome_proba = self.model_outcome.predict_proba(X)[0]
        
        # Get over/under 2.5 prediction
        over_pred = self.model_over.predict(X)[0]
        over_proba = self.model_over.predict_proba(X)[0]
        
        return {
            "fixture": fixture_data["Fixture"],
            "date": fixture_data["Date"],
            "time": fixture_data["Time"],
            "venue": fixture_data["Venue"],
            "predicted_outcome": self.outcome_map[outcome_pred],
            "outcome_confidence": max(outcome_proba),
            "outcome_probabilities": {
                "Home Win": outcome_proba[0],
                "Draw": outcome_proba[1],
                "Away Win": outcome_proba[2]
            },
            "over_25_prediction": "Over 2.5" if over_pred else "Under 2.5",
            "over_25_confidence": max(over_proba),
            "over_25_probabilities": {
                "Under 2.5": over_proba[0],
                "Over 2.5": over_proba[1]
            },
            "features_used": features
        }
    
    async def predict_all_fixtures(self) -> List[Dict]:
        """Get fixtures from agent and predict outcomes for all"""
        print("🔍 Fetching fixtures from football stats agent...")
        fixtures = await self.agent.get_fixtures()
        
        if not fixtures:
            print("❌ No fixtures found!")
            return []
        
        print(f"📊 Found {len(fixtures)} fixtures. Analyzing...")
        
        # Analyze fixtures to get detailed stats
        analyzed_fixtures = await self.agent.analyze_fixtures()
        
        predictions = []
        for fixture_data in analyzed_fixtures:
            if "Fixture" not in fixture_data and "Home" in fixture_data and "Away" in fixture_data:
                fixture_data["Fixture"] = f"{fixture_data['Home']} vs {fixture_data['Away']}"
            if not (isinstance(fixture_data, dict) and "Fixture" in fixture_data):
                print("Skipping invalid fixture_data:", fixture_data)
                continue
            try:
                prediction = self.predict_fixture(fixture_data)
                predictions.append(prediction)
                print(f"✅ Predicted: {prediction['fixture']} - {prediction['predicted_outcome']} ({prediction['outcome_confidence']:.2%})")
            except Exception as e:
                print(f"❌ Error predicting {fixture_data.get('Fixture', 'Unknown')}: {e}")
        
        return predictions
    
    def display_predictions(self, predictions: List[Dict]):
        """Display predictions in a formatted table"""
        if not predictions:
            print("No predictions to display")
            return
        
        print("\n" + "="*100)
        print("🎯 FOOTBALL FIXTURE PREDICTIONS")
        print("="*100)
        
        for pred in predictions:
            print(f"\n📅 {pred['date']} at {pred['time']}")
            print(f"🏟️  {pred['fixture']} at {pred['venue']}")
            print(f"🎲 Predicted Outcome: {pred['predicted_outcome']} (Confidence: {pred['outcome_confidence']:.1%})")
            print(f"⚽ Goals Prediction: {pred['over_25_prediction']} (Confidence: {pred['over_25_confidence']:.1%})")
            
            print("📊 Outcome Probabilities:")
            for outcome, prob in pred['outcome_probabilities'].items():
                print(f"   {outcome}: {prob:.1%}")
            
            print("📊 Goals Probabilities:")
            for goals, prob in pred['over_25_probabilities'].items():
                print(f"   {goals}: {prob:.1%}")
            
            print("-" * 80)

@app.get("/api/predictions")
async def get_predictions():
    try:
        predictor = FixturePredictor()
        predictions = await predictor.predict_all_fixtures()
        # Only return fixture, predicted_outcome, and confidence
        return [
            {
                "fixture": p["fixture"],
                "predicted_outcome": p["predicted_outcome"],
                "confidence": p["outcome_confidence"],
                "over_under": p["over_25_prediction"],
                "over_under_confidence": p["over_25_confidence"]
            }
            for p in predictions
        ]
    except Exception as e:
        return {"error": str(e)}

async def main():
    predictor = FixturePredictor()
    
    print("🚀 Starting fixture prediction analysis...")
    predictions = await predictor.predict_all_fixtures()
    
    if predictions:
        predictor.display_predictions(predictions)
        
        # Save predictions to CSV
        df = pd.DataFrame(predictions)
        df.to_csv("fixture_predictions.csv", index=False)
        print(f"\n💾 Predictions saved to fixture_predictions.csv")
    else:
        print("❌ No predictions generated")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 