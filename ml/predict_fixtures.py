import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from recommendation_engine.football_stats_agent import FootballStatsAgent
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImprovedFixturePredictor:
    def __init__(self):
        self.model_outcome = joblib.load("model_outcome.pkl")
        self.model_over = joblib.load("model_over25.pkl")
        self.agent = FootballStatsAgent()
        # Only use these features
        self.h2h_numeric_features = [
            "Home_Last6_Matches", "Home_Last6_Goals", "Home_Last6_per game", "Home_Last6_Wins",
            "Home_Last6_Draws", "Home_Last6_Losses", "Home_Last6_Over 2.5", "Home_Last6_Over 1.5", "Home_Last6_CS", "Home_Last6_BTTS",
            "Home_Overall_Matches", "Home_Overall_Goals", "Home_Overall_per game", "Home_Overall_Wins", "Home_Overall_Draws",
            "Home_Overall_Losses", "Home_Overall_Over 2.5", "Home_Overall_Over 1.5", "Home_Overall_CS", "Home_Overall_BTTS",
            "Away_Last6_Matches", "Away_Last6_Goals", "Away_Last6_per game", "Away_Last6_Wins",
            "Away_Last6_Draws", "Away_Last6_Losses", "Away_Last6_Over 2.5", "Away_Last6_Over 1.5", "Away_Last6_CS", "Away_Last6_BTTS",
            "Away_Overall_Matches", "Away_Overall_Goals", "Away_Overall_per game", "Away_Overall_Wins", "Away_Overall_Draws",
            "Away_Overall_Losses", "Away_Overall_Over 2.5", "Away_Overall_Over 1.5", "Away_Overall_CS", "Away_Overall_BTTS",
            "Home_Form_W", "Home_Form_D", "Home_Form_L", "Away_Form_W", "Away_Form_D", "Away_Form_L"
        ]
        self.h2h_feature_list = self.h2h_numeric_features  # <-- Add this line

        # Model performance metrics (from training)
        self.model_metrics = {
            "outcome_accuracy": 0.605,
            "over_under_accuracy": 0.726,
        }

    def extract_enhanced_features(self, fixture_data: Dict) -> Dict:
        """Extract only the allowed H2H features for ML prediction"""
        return {key: fixture_data.get(key, 0) for key in self.h2h_feature_list}

    def calculate_confidence_score(self, probabilities: np.ndarray, model_accuracy: float) -> float:
        max_prob = np.max(probabilities)
        prob_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = -np.log(1.0 / len(probabilities))
        normalized_entropy = prob_entropy / max_entropy
        confidence = (max_prob * 0.7 + (1 - normalized_entropy) * 0.3) * model_accuracy
        return min(confidence, 1.0)

    def predict_fixture(self, fixture_data: Dict) -> Dict:
        features = self.extract_enhanced_features(fixture_data)
        feature_values = [features.get(feature, 0) for feature in self.h2h_feature_list]
        X = np.array([feature_values])
        X_df = pd.DataFrame(X, columns=self.h2h_numeric_features)

        outcome_pred = self.model_outcome.predict(X_df)[0]
        outcome_proba = self.model_outcome.predict_proba(X_df)[0]
        over_pred = self.model_over.predict(X_df)[0]
        over_proba = self.model_over.predict_proba(X_df)[0]

        home_team = fixture_data.get('Home', 'Unknown')
        away_team = fixture_data.get('Away', 'Unknown')
        fixture_string = f"{home_team} vs {away_team}"

        # Outcome mapping
        binary_outcome_map = {0: "Home Win/Draw", 1: "Away Win"}
        outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        if len(outcome_proba) == 2:
            predicted_outcome = binary_outcome_map[outcome_pred]
            outcome_probabilities = {
                binary_outcome_map[0]: outcome_proba[0],
                binary_outcome_map[1]: outcome_proba[1]
            }
        else:
            predicted_outcome = outcome_map.get(outcome_pred, "Unknown")
            outcome_probabilities = {
                "Home Win": outcome_proba[0] if len(outcome_proba) > 0 else 0,
                "Draw": outcome_proba[1] if len(outcome_proba) > 1 else 0,
                "Away Win": outcome_proba[2] if len(outcome_proba) > 2 else 0
            }

        outcome_confidence = self.calculate_confidence_score(outcome_proba, self.model_metrics["outcome_accuracy"])
        over_under_confidence = self.calculate_confidence_score(over_proba, self.model_metrics["over_under_accuracy"])
        insights = self.generate_prediction_insights(features, outcome_proba, over_proba)

        return {
            "fixture": fixture_string,
            "date": fixture_data.get("Date", "Unknown"),
            "time": fixture_data.get("Time", "Unknown"),
            "venue": fixture_data.get("Venue", "Unknown"),
            "predicted_outcome": predicted_outcome,
            "outcome_confidence": outcome_confidence,
            "outcome_probabilities": outcome_probabilities,
            "over_25_prediction": "Over 2.5" if over_pred else "Under 2.5",
            "over_25_confidence": over_under_confidence,
            "over_25_probabilities": {
                "Under 2.5": over_proba[0],
                "Over 2.5": over_proba[1]
            },
            "features_used": features,
            # Only include H2H stats if needed for display
            "insights": insights
        }

    def generate_prediction_insights(self, features: Dict, outcome_proba: np.ndarray, over_proba: np.ndarray) -> List[str]:
        insights = []
        home_form = features.get("Home_Form", "")
        away_form = features.get("Away_Form", "")
        home_goals = float(features.get("Home_Last6_Goals", 0))
        away_goals = float(features.get("Away_Last6_Goals", 0))

        if home_form and away_form:
            if home_form.count("W") > away_form.count("W"):
                insights.append("Home team has better recent form")
            elif away_form.count("W") > home_form.count("W"):
                insights.append("Away team has better recent form")

        if home_goals > 2.5 or away_goals > 2.5:
            insights.append("High recent goal average - possible for Over 2.5")
        elif home_goals < 1.0 and away_goals < 1.0:
            insights.append("Low recent goal average - possible for Under 2.5")

        max_outcome_prob = max(outcome_proba)
        max_over_prob = max(over_proba)

        if max_outcome_prob > 0.7:
            insights.append("High confidence in outcome prediction")
        elif max_outcome_prob < 0.4:
            insights.append("Low confidence - match outcome is uncertain")

        if max_over_prob > 0.75:
            insights.append("Very high confidence in goals prediction")

        return insights
    
    async def predict_all_fixtures(self) -> List[Dict]:
        """Get fixtures from agent and predict outcomes for all"""
        print("🔍 Fetching fixtures from football stats agent...")
        fixtures = await self.agent.get_fixtures()
        
        if not fixtures:
            print("❌ No fixtures found!")
            return []
        
        print(f"📊 Found {len(fixtures)} fixtures. Analyzing with improved models...")
        
        # Analyze fixtures to get detailed stats
        analyzed_fixtures = await self.agent.analyze_fixture(fixtures)
        
        predictions = []
        for fixture_data in analyzed_fixtures:
            if "Fixture" not in fixture_data and "Home" in fixture_data and "Away" in fixture_data:
                fixture_data["Fixture"] = f"{fixture_data['Home']} vs {fixture_data['Away']}"
            if not (isinstance(fixture_data, dict) and "Fixture" in fixture_data):
                print("Skipping invalid fixture_data:", fixture_data)
                continue
            try:
                # Print the full stats gathered for this fixture
                print(f"\n--- Stats for {fixture_data['Fixture']} ---")
                for k, v in fixture_data.items():
                    print(f"{k}: {v}")
                print("--- End of stats ---\n")
                prediction = self.predict_fixture(fixture_data)
                predictions.append(prediction)
                print(f"✅ Predicted: {prediction['fixture']} - {prediction['predicted_outcome']} ({prediction['outcome_confidence']:.2%})")
                print(f"   Goals: {prediction['over_25_prediction']} ({prediction['over_25_confidence']:.2%})")
            except Exception as e:
                print(f"❌ Error predicting {fixture_data.get('Fixture', 'Unknown')}: {e}")
        
        return predictions
    
    def display_predictions(self, predictions: List[Dict]):
        """Display predictions in a formatted table with enhanced information"""
        if not predictions:
            print("No predictions to display")
            return
        
        print("\n" + "="*120)
        print("🎯 IMPROVED FOOTBALL FIXTURE PREDICTIONS")
        print("="*120)
        
        for pred in predictions:
            print(f"\n📅 {pred['date']} at {pred['time']}")
            print(f"🏟️  {pred['fixture']} at {pred['venue']}")
            print(f"🎲 Predicted Outcome: {pred['predicted_outcome']} (Confidence: {pred['outcome_confidence']:.1%})")
            print(f"⚽ Goals Prediction: {pred['over_25_prediction']} (Confidence: {pred['over_25_confidence']:.1%})")
            print(f"🤝 H2H: W {pred.get('h2h_wins', 0)} D {pred.get('h2h_draws', 0)} L {pred.get('h2h_losses', 0)} | Avg Goals (3): {pred.get('h2h_avg_goals_3', 0)} | (4): {pred.get('h2h_avg_goals_4', 0)}")
            print("📊 Outcome Probabilities:")
            for outcome, prob in pred['outcome_probabilities'].items():
                print(f"   {outcome}: {prob:.1%}")
            print("📊 Goals Probabilities:")
            for goals, prob in pred['over_25_probabilities'].items():
                print(f"   {goals}: {prob:.1%}")
            if pred.get('insights'):
                print("💡 Insights:")
                for insight in pred['insights']:
                    print(f"   • {insight}")
            print("-" * 80)

@app.get("/api/predictions")
async def get_predictions():
    try:
        predictor = ImprovedFixturePredictor()
        predictions = await predictor.predict_all_fixtures()
        print(predictions)
        # Return enhanced prediction data
        return [
            {
                "fixture": p["fixture"],
                "predicted_outcome": p["predicted_outcome"],
                "confidence": p["outcome_confidence"],
                "over_under": p["over_25_prediction"],
                "over_under_confidence": p["over_25_confidence"],
                "insights": p.get("insights", []),
                "model_accuracy": p.get("model_accuracy", ""),
                # New H2H and Elo fields
                "h2h_wins": p.get("h2h_wins", 0),
                "h2h_draws": p.get("h2h_draws", 0),
                "h2h_losses": p.get("h2h_losses", 0),
               
            }
            for p in predictions
        ]
    
    except Exception as e:
        return {"error": str(e)}

async def main():
    predictor = ImprovedFixturePredictor()
    
    print("🚀 Starting improved fixture prediction analysis...")
    predictions = await predictor.predict_all_fixtures()
    
    if predictions:
        predictor.display_predictions(predictions)
        
        # Save predictions to CSV
        df = pd.DataFrame(predictions)
        df.to_csv("improved_fixture_predictions.csv", index=False)
        print(f"\n💾 Improved predictions saved to improved_fixture_predictions.csv")
    else:
        print("❌ No predictions generated")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)