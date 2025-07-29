from flask import Flask, jsonify, request
import requests
import random
from datetime import datetime, timedelta
import statistics
from typing import List, Dict

app = Flask(__name__)

class BettingSuggestion:
    def __init__(self, match_id, home_team, away_team, prediction, confidence, reasoning, odds):
        self.match_id = match_id
        self.home_team = home_team
        self.away_team = away_team
        self.prediction = prediction
        self.confidence = confidence
        self.reasoning = reasoning
        self.odds = odds
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "odds": self.odds,
            "timestamp": self.timestamp
        }

class MatchAnalyzer:
    def __init__(self):
        self.match_data_url = "http://localhost:4002/api/matches/recommendation-data"
        self.predictions = ["Home Win", "Away Win", "Draw", "Over 2.5 Goals", "Under 2.5 Goals"]
        self.reasoning_templates = {
            "Home Win": [
                "Strong home form with {home_wins} wins in last 3 matches",
                "Home team scoring average of {home_goals} goals per game",
                "Home team has won {head2head_home_wins} of last 3 head-to-head matches"
            ],
            "Away Win": [
                "Strong away form with {away_wins} wins in last 3 matches",
                "Away team scoring average of {away_goals} goals per game",
                "Away team has won {head2head_away_wins} of last 3 head-to-head matches"
            ],
            "Draw": [
                "Both teams in similar form",
                "Historical tendency for draws in this fixture",
                "Both teams have similar goal scoring records"
            ],
            "Over 2.5 Goals": [
                "High scoring teams with combined average of {total_goals} goals per game",
                "Both teams have strong attacking records",
                "Historical tendency for high scoring matches"
            ],
            "Under 2.5 Goals": [
                "Strong defensive records from both teams",
                "Low scoring teams with combined average of {total_goals} goals per game",
                "Historical tendency for low scoring matches"
            ]
        }

    def analyze_team_form(self, matches: List[Dict]) -> Dict:
        if not matches:
            return {"wins": 0, "goals": 0}
        
        wins = 0
        goals = 0
        for match in matches:
            if match["score"]["fullTime"]["home"] is not None:
                if match["homeTeam"]["id"] == matches[0]["homeTeam"]["id"]:
                    goals += match["score"]["fullTime"]["home"]
                    if match["score"]["fullTime"]["home"] > match["score"]["fullTime"]["away"]:
                        wins += 1
                else:
                    goals += match["score"]["fullTime"]["away"]
                    if match["score"]["fullTime"]["away"] > match["score"]["fullTime"]["home"]:
                        wins += 1
        
        return {
            "wins": wins,
            "goals": goals / len(matches) if matches else 0
        }

    def analyze_head_to_head(self, matches: List[Dict], team_id: str) -> Dict:
        if not matches:
            return {"wins": 0}
        
        wins = 0
        for match in matches:
            if match["score"]["fullTime"]["home"] is not None:
                if match["homeTeam"]["id"] == team_id:
                    if match["score"]["fullTime"]["home"] > match["score"]["fullTime"]["away"]:
                        wins += 1
                else:
                    if match["score"]["fullTime"]["away"] > match["score"]["fullTime"]["home"]:
                        wins += 1
        
        return {"wins": wins}

    def calculate_confidence(self, home_form: Dict, away_form: Dict, 
                           head2head_home: Dict, head2head_away: Dict) -> float:
        # Base confidence on team form and head-to-head record
        home_strength = (home_form["wins"] * 0.4 + home_form["goals"] * 0.3 + head2head_home["wins"] * 0.3)
        away_strength = (away_form["wins"] * 0.4 + away_form["goals"] * 0.3 + head2head_away["wins"] * 0.3)
        
        # Normalize confidence to 0-1 range
        total_strength = home_strength + away_strength
        if total_strength == 0:
            return 0.5
        
        return max(0.5, min(0.95, (home_strength / total_strength)))

    def generate_suggestions(self) -> List[BettingSuggestion]:
        try:
            response = requests.get(self.match_data_url)
            data = response.json()
            
            suggestions = []
            for match in data["upcomingMatches"]:
                home_team = match["homeTeam"]
                away_team = match["awayTeam"]
                
                # Get team form
                home_form = self.analyze_team_form(data["headToHeadData"].get(home_team["id"], []))
                away_form = self.analyze_team_form(data["headToHeadData"].get(away_team["id"], []))
                
                # Get head-to-head data
                head2head_home = self.analyze_head_to_head(data["headToHeadData"].get(home_team["id"], []), home_team["id"])
                head2head_away = self.analyze_head_to_head(data["headToHeadData"].get(away_team["id"], []), away_team["id"])
                
                # Calculate confidence
                confidence = self.calculate_confidence(home_form, away_form, head2head_home, head2head_away)
                
                # Determine prediction
                if home_form["wins"] > away_form["wins"] and head2head_home["wins"] > head2head_away["wins"]:
                    prediction = "Home Win"
                elif away_form["wins"] > home_form["wins"] and head2head_away["wins"] > head2head_home["wins"]:
                    prediction = "Away Win"
                else:
                    prediction = "Draw"
                
                # Generate reasoning
                reasoning = random.choice(self.reasoning_templates[prediction]).format(
                    home_wins=home_form["wins"],
                    away_wins=away_form["wins"],
                    home_goals=round(home_form["goals"], 1),
                    away_goals=round(away_form["goals"], 1),
                    head2head_home_wins=head2head_home["wins"],
                    head2head_away_wins=head2head_away["wins"],
                    total_goals=round(home_form["goals"] + away_form["goals"], 1)
                )
                
                # Calculate odds (simplified)
                odds = round(1 / confidence, 2)
                
                suggestion = BettingSuggestion(
                    match["id"],
                    home_team["name"],
                    away_team["name"],
                    prediction,
                    confidence,
                    reasoning,
                    odds
                )
                suggestions.append(suggestion)
            
            # Sort by confidence and return top 25
            suggestions.sort(key=lambda x: x.confidence, reverse=True)
            return suggestions[:25]
            
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            return []

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    analyzer = MatchAnalyzer()
    suggestions = analyzer.generate_suggestions()
    return jsonify([s.to_dict() for s in suggestions])

@app.route('/api/suggestions/random', methods=['GET'])
def get_random_suggestion():
    analyzer = MatchAnalyzer()
    suggestions = analyzer.generate_suggestions()
    return jsonify(random.choice(suggestions).to_dict())

@app.route('/api/suggestions/match/<match_id>', methods=['GET'])
def get_suggestion_by_match(match_id):
    analyzer = MatchAnalyzer()
    suggestions = analyzer.generate_suggestions()
    suggestion = next((s for s in suggestions if s.match_id == match_id), None)
    if suggestion:
        return jsonify(suggestion.to_dict())
    return jsonify({"error": "Suggestion not found"}), 404

@app.route('/api/suggestions/confidence/<float:min_confidence>', methods=['GET'])
def get_suggestions_by_confidence(min_confidence):
    analyzer = MatchAnalyzer()
    suggestions = analyzer.generate_suggestions()
    filtered_suggestions = [s for s in suggestions if s.confidence >= min_confidence]
    return jsonify([s.to_dict() for s in filtered_suggestions])

@app.route('/api/suggestions/team/<team_name>', methods=['GET'])
def get_suggestions_by_team(team_name):
    analyzer = MatchAnalyzer()
    suggestions = analyzer.generate_suggestions()
    team_suggestions = [
        s for s in suggestions 
        if team_name.lower() in s.home_team.lower() or team_name.lower() in s.away_team.lower()
    ]
    return jsonify([s.to_dict() for s in team_suggestions])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)