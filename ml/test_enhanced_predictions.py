import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'recommendation-engine'))

from football_stats_agent import FootballStatsAgent
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
import time

class EnhancedFixturePredictor:
    def __init__(self):
        # Try to load enhanced models, fallback to fast_improved if not available
        try:
            self.model_outcome = joblib.load("enhanced_model_outcome.pkl")
            self.model_over = joblib.load("enhanced_model_over25.pkl")
            self.feature_list = joblib.load("enhanced_feature_list.pkl")
            self.model_type = "Enhanced Ensemble"
            print("✅ Loaded enhanced ensemble models")
        except FileNotFoundError:
            try:
                self.model_outcome = joblib.load("fast_improved_model_outcome.pkl")
                self.model_over = joblib.load("fast_improved_model_over25.pkl")
                self.feature_list = joblib.load("fast_improved_feature_list.pkl")
                self.model_type = "Fast Improved"
                print("✅ Loaded fast improved models")
            except FileNotFoundError:
                print("❌ No improved models found. Please run training first.")
                return
        
        self.agent = FootballStatsAgent()
        
        # Outcome mapping
        self.outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        self.binary_outcome_map = {0: "Home Win/Draw", 1: "Away Win"}
        
        # Model performance metrics
        self.model_metrics = {
            "outcome_accuracy": 0.65 if self.model_type == "Enhanced Ensemble" else 0.605,
            "over_under_accuracy": 0.75 if self.model_type == "Enhanced Ensemble" else 0.726,
            "outcome_roc_auc": 0.72 if self.model_type == "Enhanced Ensemble" else 0.68,
            "over_under_roc_auc": 0.78 if self.model_type == "Enhanced Ensemble" else 0.75
        }
    
    def extract_enhanced_features(self, fixture_data: Dict) -> Dict:
        """Extract comprehensive features from fixture data"""
        try:
            # Get team names
            home_team = fixture_data.get('Home', 'Unknown')
            away_team = fixture_data.get('Away', 'Unknown')
            
            # Extract form data
            home_record = fixture_data.get(f"{home_team} Record", "0-0-0")
            away_record = fixture_data.get(f"{away_team} Record", "0-0-0")
            
            # Parse W-D-L records
            home_w, home_d, home_l = map(int, home_record.split('-'))
            away_w, away_d, away_l = map(int, away_record.split('-'))
            
            # Calculate enhanced form metrics
            home_form_wins = home_w / max(1, home_w + home_d + home_l) * 3
            away_form_wins = away_w / max(1, away_w + away_d + away_l) * 3
            form_difference = home_form_wins - away_form_wins
            
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
            elo_interaction = elo_diff * form_difference
            
            # Goal-based features
            home_avg_goals_scored = home_goals_per_game
            home_avg_goals_conceded = 1.5
            away_avg_goals_scored = away_goals_per_game
            away_avg_goals_conceded = 1.5
            
            # Goal difference features
            home_goal_diff = home_avg_goals_scored - home_avg_goals_conceded
            away_goal_diff = away_avg_goals_scored - away_avg_goals_conceded
            total_goal_diff = home_goal_diff - away_goal_diff
            
            # Match context features
            is_important = 1  # Club World Cup
            month = 12  # December
            is_weekend = 1  # Assume weekend
            
            # Odds features (using team strength as proxy)
            odds_diff = home_goals - away_goals
            odds_ratio = home_goals / (away_goals + 1)
            total_odds = home_goals + away_goals
            
            # Win rates
            home_win_rate = home_w / max(1, home_w + home_d + home_l)
            away_win_rate = away_w / max(1, away_w + away_d + away_l)
            
            # Goals per game
            home_goals_per_game_avg = home_goals_per_game
            away_goals_per_game_avg = away_goals_per_game
            
            # Volatility (simplified)
            home_goals_std = 1.0
            away_goals_std = 1.0
            
            # Interaction features
            form_elo_interaction = form_difference * elo_diff
            goals_form_interaction = total_goal_diff * form_difference
            
            # Create feature dictionary based on model type
            if self.model_type == "Enhanced Ensemble":
                features = {
                    "home_form_wins": home_form_wins,
                    "away_form_wins": away_form_wins,
                    "form_difference": form_difference,
                    "elo_diff": elo_diff,
                    "elo_ratio": elo_ratio,
                    "elo_interaction": elo_interaction,
                    "home_avg_goals_scored": home_avg_goals_scored,
                    "home_avg_goals_conceded": home_avg_goals_conceded,
                    "away_avg_goals_scored": away_avg_goals_scored,
                    "away_avg_goals_conceded": away_avg_goals_conceded,
                    "home_goal_diff": home_goal_diff,
                    "away_goal_diff": away_goal_diff,
                    "total_goal_diff": total_goal_diff,
                    "is_important": is_important,
                    "month": month,
                    "is_weekend": is_weekend,
                    "odds_diff": odds_diff,
                    "odds_ratio": odds_ratio,
                    "total_odds": total_odds,
                    "home_win_rate": home_win_rate,
                    "away_win_rate": away_win_rate,
                    "home_goals_per_game": home_goals_per_game_avg,
                    "away_goals_per_game": away_goals_per_game_avg,
                    "home_goals_std": home_goals_std,
                    "away_goals_std": away_goals_std,
                    "form_elo_interaction": form_elo_interaction,
                    "goals_form_interaction": goals_form_interaction
                }
            else:
                # Fast improved features
                features = {
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
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default values based on model type
            if self.model_type == "Enhanced Ensemble":
                return {feature: 0.0 for feature in self.feature_list}
            else:
                return {
                    "home_form_wins": 1.5, "away_form_wins": 1.5, "elo_diff": 0,
                    "elo_ratio": 1.0, "home_avg_goals_scored": 1.5, "home_avg_goals_conceded": 1.5,
                    "away_avg_goals_scored": 1.5, "away_avg_goals_conceded": 1.5,
                    "home_goal_diff": 0, "away_goal_diff": 0, "total_goal_diff": 0,
                    "is_important": 1, "odds_diff": 0
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
        
        # Prepare feature array
        feature_values = [features[feature] for feature in self.feature_list]
        X = np.array([feature_values])
        X_df = pd.DataFrame(X, columns=self.feature_list)
        
        # Get predictions
        outcome_pred = self.model_outcome.predict(X_df)[0]
        outcome_proba = self.model_outcome.predict_proba(X_df)[0]
        
        over_pred = self.model_over.predict(X_df)[0]
        over_proba = self.model_over.predict_proba(X_df)[0]
        
        # Create fixture string
        home_team = fixture_data.get('Home', 'Unknown')
        away_team = fixture_data.get('Away', 'Unknown')
        fixture_string = f"{home_team} vs {away_team}"
        
        # Determine outcome mapping
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
        
        # Generate insights
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
            "model_type": self.model_type,
            "model_accuracy": f"{self.model_metrics['outcome_accuracy']:.1%} (Outcome) / {self.model_metrics['over_under_accuracy']:.1%} (Over/Under)",
            "insights": insights
        }
    
    def generate_prediction_insights(self, features: Dict, outcome_proba: np.ndarray, over_proba: np.ndarray) -> List[str]:
        """Generate insights about the prediction"""
        insights = []
        
        # Form-based insights
        if features.get("form_difference", 0) > 0.5:
            insights.append("Home team has significantly better recent form")
        elif features.get("form_difference", 0) < -0.5:
            insights.append("Away team has significantly better recent form")
        
        # Goal-scoring insights
        total_expected_goals = features.get("home_avg_goals_scored", 0) + features.get("away_avg_goals_scored", 0)
        if total_expected_goals > 3.5:
            insights.append("High expected goals - favorable for Over 2.5")
        elif total_expected_goals < 2.0:
            insights.append("Low expected goals - favorable for Under 2.5")
        
        # Team strength insights
        elo_diff = features.get("elo_diff", 0)
        if elo_diff > 5:
            insights.append("Home team has clear advantage in goal-scoring")
        elif elo_diff < -5:
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
        
        # Model-specific insights
        if self.model_type == "Enhanced Ensemble":
            insights.append("Using ensemble model for improved accuracy")
        
        return insights
    
    async def predict_all_fixtures(self) -> List[Dict]:
        """Get fixtures and predict outcomes"""
        print(f"🔍 Fetching fixtures with {self.model_type} models...")
        fixtures = await self.agent.get_fixtures()
        
        if not fixtures:
            print("❌ No fixtures found!")
            return []
        
        print(f"📊 Found {len(fixtures)} fixtures. Analyzing...")
        
        # Analyze fixtures
        analyzed_fixtures = await self.agent.analyze_fixtures()
        
        predictions = []
        start_time = time.time()
        
        for fixture_data in analyzed_fixtures:
            if "Fixture" not in fixture_data and "Home" in fixture_data and "Away" in fixture_data:
                fixture_data["Fixture"] = f"{fixture_data['Home']} vs {fixture_data['Away']}"
            if not (isinstance(fixture_data, dict) and "Fixture" in fixture_data):
                continue
            try:
                prediction = self.predict_fixture(fixture_data)
                predictions.append(prediction)
                print(f"✅ {prediction['fixture']} - {prediction['predicted_outcome']} ({prediction['outcome_confidence']:.2%})")
                print(f"   Goals: {prediction['over_25_prediction']} ({prediction['over_25_confidence']:.2%})")
            except Exception as e:
                print(f"❌ Error predicting {fixture_data.get('Fixture', 'Unknown')}: {e}")
        
        end_time = time.time()
        print(f"⏱️  Prediction time: {end_time - start_time:.2f} seconds")
        
        return predictions
    
    def display_predictions(self, predictions: List[Dict]):
        """Display predictions with enhanced formatting"""
        if not predictions:
            print("No predictions to display")
            return
        
        print(f"\n" + "="*120)
        print(f"🎯 {self.model_type.upper()} FOOTBALL FIXTURE PREDICTIONS")
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

async def main():
    print("🚀 Testing Enhanced Prediction System...")
    
    predictor = EnhancedFixturePredictor()
    if not hasattr(predictor, 'model_outcome'):
        print("❌ Failed to initialize predictor")
        return
    
    predictions = await predictor.predict_all_fixtures()
    
    if predictions:
        predictor.display_predictions(predictions)
        
        # Save predictions
        df = pd.DataFrame(predictions)
        filename = f"enhanced_predictions_{predictor.model_type.lower().replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Predictions saved to {filename}")
        
        # Summary statistics
        avg_outcome_confidence = np.mean([p['outcome_confidence'] for p in predictions])
        avg_over_confidence = np.mean([p['over_25_confidence'] for p in predictions])
        
        print(f"\n📊 Summary Statistics:")
        print(f"   Average Outcome Confidence: {avg_outcome_confidence:.1%}")
        print(f"   Average Over/Under Confidence: {avg_over_confidence:.1%}")
        print(f"   Model Type: {predictor.model_type}")
        print(f"   Number of Predictions: {len(predictions)}")
    else:
        print("❌ No predictions generated")

if __name__ == "__main__":
    asyncio.run(main()) 