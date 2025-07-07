import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'recommendation-engine'))

from football_stats_agent import FootballStatsAgent
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
        # Load the improved models
        self.model_outcome = joblib.load("fast_improved_model_outcome.pkl")
        self.model_over = joblib.load("fast_improved_model_over25.pkl")
        self.feature_list = joblib.load("fast_improved_feature_list.pkl")
        self.agent = FootballStatsAgent()
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Outcome mapping for different model types
        self.outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        self.binary_outcome_map = {0: "Home Win/Draw", 1: "Away Win"}
        
        # Model performance metrics (from training)
        self.model_metrics = {
            "outcome_accuracy": 0.605,
            "over_under_accuracy": 0.726,
            "outcome_roc_auc": 0.68,
            "over_under_roc_auc": 0.75
        }
    
    def extract_enhanced_features(self, fixture_data: Dict) -> Dict:
        """Extract comprehensive features from fixture data for ML prediction"""
        try:
            # Get team names
            home_team = fixture_data.get('Home', 'Unknown')
            away_team = fixture_data.get('Away', 'Unknown')
            
            # Extract form data from team records
            home_record = fixture_data.get(f"{home_team} Record", "0-0-0")
            away_record = fixture_data.get(f"{away_team} Record", "0-0-0")
            
            # Parse W-D-L records
            home_w, home_d, home_l = map(int, home_record.split('-'))
            away_w, away_d, away_l = map(int, away_record.split('-'))
            
            # Calculate enhanced form metrics
            home_form_wins = home_w / max(1, home_w + home_d + home_l) * 3
            away_form_wins = away_w / max(1, away_w + away_d + away_l) * 3
            
            # Get team stats
            home_goals = fixture_data.get(f"{home_team} Goals", 0)
            away_goals = fixture_data.get(f"{away_team} Goals", 0)
            home_goals_per_game = fixture_data.get(f"{home_team} Goals/Game", 0)
            away_goals_per_game = fixture_data.get(f"{away_team} Goals/Game", 0)
            home_xg = fixture_data.get(f"{home_team} xG", 0)
            away_xg = fixture_data.get(f"{away_team} xG", 0)
            
            # Enhanced Elo-like features
            elo_diff = home_goals - away_goals
            elo_ratio = home_goals / (away_goals + 1)
            
            # Goal-based features with rolling averages
            home_avg_goals_scored = home_goals_per_game
            home_avg_goals_conceded = 1.5  # Default, could be enhanced with actual data
            away_avg_goals_scored = away_goals_per_game
            away_avg_goals_conceded = 1.5  # Default, could be enhanced with actual data
            
            # Goal difference features
            home_goal_diff = home_avg_goals_scored - home_avg_goals_conceded
            away_goal_diff = away_avg_goals_scored - away_avg_goals_conceded
            total_goal_diff = home_goal_diff - away_goal_diff
            
            # Match importance (Club World Cup is important)
            is_important = 1
            
            # Odds difference (using team strength as proxy)
            odds_diff = home_goals - away_goals
            
            # Additional advanced features
            home_win_rate = home_w / max(1, home_w + home_d + home_l)
            away_win_rate = away_w / max(1, away_w + away_d + away_l)
            form_difference = home_form_wins - away_form_wins
            
            # Expected goals features
            xg_difference = home_xg - away_xg
            total_expected_goals = home_xg + away_xg
            
            return {
                "home_form_wins": home_form_wins,
                "away_form_wins": away_form_wins,
                "elo_diff": elo_diff,
                "elo_ratio": elo_ratio,
                "home_avg_goals_scored": home_avg_goals_scored,
                "home_avg_goals_conceded": home_avg_goals_conceded,
                "away_avg_goals_scored": away_avg_goals_scored,
                "away_avg_goals_conceded": away_avg_goals_conceded,
                "home_goal_diff": home_goal_diff,
                "away_goal_diff": away_goal_diff,
                "total_goal_diff": total_goal_diff,
                "is_important": is_important,
                "odds_diff": odds_diff,
                # Additional features for enhanced prediction
                "home_win_rate": home_win_rate,
                "away_win_rate": away_win_rate,
                "form_difference": form_difference,
                "xg_difference": xg_difference,
                "total_expected_goals": total_expected_goals
            }
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return comprehensive default values
            return {
                "home_form_wins": 1.5,
                "away_form_wins": 1.5,
                "elo_diff": 0,
                "elo_ratio": 1.0,
                "home_avg_goals_scored": 1.5,
                "home_avg_goals_conceded": 1.5,
                "away_avg_goals_scored": 1.5,
                "away_avg_goals_conceded": 1.5,
                "home_goal_diff": 0,
                "away_goal_diff": 0,
                "total_goal_diff": 0,
                "is_important": 1,
                "odds_diff": 0,
                "home_win_rate": 0.33,
                "away_win_rate": 0.33,
                "form_difference": 0,
                "xg_difference": 0,
                "total_expected_goals": 3.0
            }
    
    def calculate_confidence_score(self, probabilities: np.ndarray, model_accuracy: float) -> float:
        """Calculate confidence score based on probability distribution and model accuracy"""
        max_prob = np.max(probabilities)
        prob_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = -np.log(1.0 / len(probabilities))
        normalized_entropy = prob_entropy / max_entropy
        
        # Combine probability confidence with model accuracy
        confidence = (max_prob * 0.7 + (1 - normalized_entropy) * 0.3) * model_accuracy
        return min(confidence, 1.0)
    
    def predict_fixture(self, fixture_data: Dict) -> Dict:
        """Make enhanced predictions for a single fixture"""
        features = self.extract_enhanced_features(fixture_data)
        
        # Prepare feature array with proper feature names
        feature_values = [features[feature] for feature in self.feature_list]
        X = np.array([feature_values])
        
        # Create DataFrame with feature names
        X_df = pd.DataFrame(X, columns=self.feature_list)
        
        # Get outcome prediction
        outcome_pred = self.model_outcome.predict(X_df)[0]
        outcome_proba = self.model_outcome.predict_proba(X_df)[0]
        
        # Get over/under 2.5 prediction
        over_pred = self.model_over.predict(X_df)[0]
        over_proba = self.model_over.predict_proba(X_df)[0]
        
        # Create fixture string
        home_team = fixture_data.get('Home', 'Unknown')
        away_team = fixture_data.get('Away', 'Unknown')
        fixture_string = f"{home_team} vs {away_team}"
        
        # Determine outcome mapping based on model type
        if len(outcome_proba) == 2:
            predicted_outcome = self.binary_outcome_map[outcome_pred]
            outcome_probabilities = {
                self.binary_outcome_map[0]: outcome_proba[0],
                self.binary_outcome_map[1]: outcome_proba[1]
            }
        else:
            predicted_outcome = self.outcome_map.get(outcome_pred, "Unknown")
            outcome_probabilities = {
                "Home Win": outcome_proba[0] if len(outcome_proba) > 0 else 0,
                "Draw": outcome_proba[1] if len(outcome_proba) > 1 else 0,
                "Away Win": outcome_proba[2] if len(outcome_proba) > 2 else 0
            }
        
        # Calculate confidence scores
        outcome_confidence = self.calculate_confidence_score(outcome_proba, self.model_metrics["outcome_accuracy"])
        over_under_confidence = self.calculate_confidence_score(over_proba, self.model_metrics["over_under_accuracy"])
        
        # Add prediction insights
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
            "model_accuracy": f"{self.model_metrics['outcome_accuracy']:.1%} (Outcome) / {self.model_metrics['over_under_accuracy']:.1%} (Over/Under)",
            "insights": insights
        }
    
    def generate_prediction_insights(self, features: Dict, outcome_proba: np.ndarray, over_proba: np.ndarray) -> List[str]:
        """Generate insights about the prediction based on features and probabilities"""
        insights = []
        
        # Form-based insights
        if features["form_difference"] > 0.5:
            insights.append("Home team has significantly better recent form")
        elif features["form_difference"] < -0.5:
            insights.append("Away team has significantly better recent form")
        
        # Goal-scoring insights
        if features["total_expected_goals"] > 3.5:
            insights.append("High expected goals - favorable for Over 2.5")
        elif features["total_expected_goals"] < 2.0:
            insights.append("Low expected goals - favorable for Under 2.5")
        
        # Team strength insights
        if features["elo_diff"] > 5:
            insights.append("Home team has clear advantage in goal-scoring")
        elif features["elo_diff"] < -5:
            insights.append("Away team has clear advantage in goal-scoring")
        
        # Confidence insights
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
        # Return enhanced prediction data
        return [
            {
                "fixture": p["fixture"],
                "predicted_outcome": p["predicted_outcome"],
                "confidence": p["outcome_confidence"],
                "over_under": p["over_25_prediction"],
                "over_under_confidence": p["over_25_confidence"],
                "insights": p.get("insights", []),
                "model_accuracy": p.get("model_accuracy", "")
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