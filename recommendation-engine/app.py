from flask import Flask, jsonify, request
import random
from datetime import datetime, timedelta

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

# Sample data - in a real application, this would be calculated using ML models
suggestions = [
    BettingSuggestion(
        "1",
        "Manchester United",
        "Liverpool",
        "Home Win",
        0.75,
        "Strong home form and key players returning from injury",
        2.10
    ),
    BettingSuggestion(
        "2",
        "Arsenal",
        "Chelsea",
        "Draw",
        0.65,
        "Both teams in good form, historically close matches",
        3.20
    ),
    BettingSuggestion(
        "3",
        "Barcelona",
        "Real Madrid",
        "Over 2.5 Goals",
        0.80,
        "High-scoring fixture historically, both teams attacking well",
        1.75
    )
]

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    return jsonify([s.to_dict() for s in suggestions])

@app.route('/api/suggestions/random', methods=['GET'])
def get_random_suggestion():
    return jsonify(random.choice(suggestions).to_dict())

@app.route('/api/suggestions/match/<match_id>', methods=['GET'])
def get_suggestion_by_match(match_id):
    suggestion = next((s for s in suggestions if s.match_id == match_id), None)
    if suggestion:
        return jsonify(suggestion.to_dict())
    return jsonify({"error": "Suggestion not found"}), 404

@app.route('/api/suggestions/confidence/<float:min_confidence>', methods=['GET'])
def get_suggestions_by_confidence(min_confidence):
    filtered_suggestions = [s for s in suggestions if s.confidence >= min_confidence]
    return jsonify([s.to_dict() for s in filtered_suggestions])

@app.route('/api/suggestions/team/<team_name>', methods=['GET'])
def get_suggestions_by_team(team_name):
    team_suggestions = [
        s for s in suggestions 
        if team_name.lower() in s.home_team.lower() or team_name.lower() in s.away_team.lower()
    ]
    return jsonify([s.to_dict() for s in team_suggestions])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)