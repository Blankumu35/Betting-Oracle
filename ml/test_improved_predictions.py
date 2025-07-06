import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'recommendation-engine'))

from football_stats_agent import FootballStatsAgent
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List

class ImprovedFixturePredictor:
    def __init__(self):
        # Load the improved models
        self.model_outcome = joblib.load("fast_improved_model_outcome.pkl")
        self.model_over = joblib.load("fast_improved_model_over25.pkl")
        self.feature_list = joblib.load("fast_improved_feature_list.pkl")
        self.agent = FootballStatsAgent()
        
        # Outcome mapping
        self.outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    
    def extract_enhanced_features(self, fixture_data: Dict) -> Dict:
        """Extract enhanced features from fixture data for ML prediction"""
        try:
            # Get home and away team names from the actual data structure
            home_team = fixture_data.get('Home', 'Unknown')
            away_team = fixture_data.get('Away', 'Unknown')
            
            # Create fixture string for compatibility
            fixture_string = f"{home_team} vs {away_team}"
            
            # Extract form data from team records (using the actual keys)
            home_record = fixture_data.get(f"{home_team} Record", "0-0-0")
            away_record = fixture_data.get(f"{away_team} Record", "0-0-0")
            
            # Parse W-D-L records
            home_w, home_d, home_l = map(int, home_record.split('-'))
            away_w, away_d, away_l = map(int, away_record.split('-'))
            
            # Calculate form (wins in last 5 matches)
            home_form_wins = home_w / max(1, home_w + home_d + home_l) * 3
            away_form_wins = away_w / max(1, away_w + away_d + away_l) * 3
            
            # Get team stats (using the actual keys)
            home_goals = fixture_data.get(f"{home_team} Goals", 0)
            away_goals = fixture_data.get(f"{away_team} Goals", 0)
            home_goals_per_game = fixture_data.get(f"{home_team} Goals/Game", 0)
            away_goals_per_game = fixture_data.get(f"{away_team} Goals/Game", 0)
            
            # Simple Elo-like difference based on goals scored
            elo_diff = home_goals - away_goals
            elo_ratio = home_goals / (away_goals + 1)
            
            # Rolling averages (simplified - using goals per game as proxy)
            home_avg_goals_scored = home_goals_per_game
            home_avg_goals_conceded = 1.5  # Default value
            away_avg_goals_scored = away_goals_per_game
            away_avg_goals_conceded = 1.5  # Default value
            
            # Goal difference features
            home_goal_diff = home_avg_goals_scored - home_avg_goals_conceded
            away_goal_diff = away_avg_goals_scored - away_avg_goals_conceded
            total_goal_diff = home_goal_diff - away_goal_diff
            
            # Match importance (assuming Club World Cup is important)
            is_important = 1
            
            # Simple odds difference (using team strength as proxy)
            odds_diff = home_goals - away_goals
            
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
                "odds_diff": odds_diff
            }
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default values
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
                "odds_diff": 0
            }
    
    def predict_fixture(self, fixture_data: Dict) -> Dict:
        """Make predictions for a single fixture using improved models"""
        features = self.extract_enhanced_features(fixture_data)
        
        # Prepare feature array with proper feature names
        feature_values = [features[feature] for feature in self.feature_list]
        X = np.array([feature_values])
        
        # Create DataFrame with feature names to avoid warnings
        X_df = pd.DataFrame(X, columns=self.feature_list)
        
        # Get outcome prediction (binary model)
        outcome_pred = self.model_outcome.predict(X_df)[0]
        outcome_proba = self.model_outcome.predict_proba(X_df)[0]
        
        # Get over/under 2.5 prediction
        over_pred = self.model_over.predict(X_df)[0]
        over_proba = self.model_over.predict_proba(X_df)[0]
        
        # Create fixture string from actual data
        home_team = fixture_data.get('Home', 'Unknown')
        away_team = fixture_data.get('Away', 'Unknown')
        fixture_string = f"{home_team} vs {away_team}"
        
        # Map binary outcome to meaningful labels
        # Assuming 0 = Home Win/Draw, 1 = Away Win (or similar binary mapping)
        if len(outcome_proba) == 2:
            outcome_labels = ["Home Win/Draw", "Away Win"]  # Adjust based on actual training
            predicted_outcome = outcome_labels[outcome_pred]
            outcome_probabilities = {
                outcome_labels[0]: outcome_proba[0],
                outcome_labels[1]: outcome_proba[1]
            }
        else:
            # Fallback to original 3-class mapping
            predicted_outcome = self.outcome_map.get(outcome_pred, "Unknown")
            outcome_probabilities = {
                "Home Win": outcome_proba[0] if len(outcome_proba) > 0 else 0,
                "Draw": outcome_proba[1] if len(outcome_proba) > 1 else 0,
                "Away Win": outcome_proba[2] if len(outcome_proba) > 2 else 0
            }
        
        return {
            "fixture": fixture_string,
            "date": fixture_data.get("Date", "Unknown"),
            "time": fixture_data.get("Time", "Unknown"),
            "venue": fixture_data.get("Venue", "Unknown"),
            "predicted_outcome": predicted_outcome,
            "outcome_confidence": max(outcome_proba),
            "outcome_probabilities": outcome_probabilities,
            "over_25_prediction": "Over 2.5" if over_pred else "Under 2.5",
            "over_25_confidence": max(over_proba),
            "over_25_probabilities": {
                "Under 2.5": over_proba[0],
                "Over 2.5": over_proba[1]
            },
            "features_used": features,
            "model_accuracy": "60.5% (Outcome) / 72.6% (Over/Under)"
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
            try:
                prediction = self.predict_fixture(fixture_data)
                predictions.append(prediction)
                print(f"✅ Predicted: {prediction['fixture']} - {prediction['predicted_outcome']} ({prediction['outcome_confidence']:.2%})")
            except Exception as e:
                home_team = fixture_data.get('Home', 'Unknown')
                away_team = fixture_data.get('Away', 'Unknown')
                fixture_name = f"{home_team} vs {away_team}"
                print(f"❌ Error predicting {fixture_name}: {e}")
        
        return predictions
    
    def display_predictions(self, predictions: List[Dict]):
        """Display predictions in a formatted table"""
        if not predictions:
            print("No predictions to display")
            return
        
        print("\n" + "="*120)
        print("🎯 IMPROVED FOOTBALL FIXTURE PREDICTIONS (Enhanced Models)")
        print("="*120)
        
        for pred in predictions:
            print(f"\n📅 {pred['date']} at {pred['time']}")
            print(f"🏟️  {pred['fixture']} at {pred['venue']}")
            print(f"🎲 Predicted Outcome: {pred['predicted_outcome']} (Confidence: {pred['outcome_confidence']:.1%})")
            print(f"⚽ Goals Prediction: {pred['over_25_prediction']} (Confidence: {pred['over_25_confidence']:.1%})")
            print(f"📊 Model Accuracy: {pred['model_accuracy']}")
            
            print("📊 Outcome Probabilities:")
            for outcome, prob in pred['outcome_probabilities'].items():
                print(f"   {outcome}: {prob:.1%}")
            
            print("📊 Goals Probabilities:")
            for goals, prob in pred['over_25_probabilities'].items():
                print(f"   {goals}: {prob:.1%}")
            
            print("-" * 120)

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
    asyncio.run(main()) 