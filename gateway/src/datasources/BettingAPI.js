const { RESTDataSource } = require('apollo-datasource-rest');

class BettingAPI extends RESTDataSource {
  constructor() {
    super();
    this.baseURL = process.env.BETTING_SERVICE_URL || 'http://localhost:4001';
  }

  async getSuggestions(userId) {
    try {
      const response = await this.get(`/suggestions/${userId}`);
      return response.map(suggestion => ({
        id: suggestion.id,
        matchId: suggestion.matchId,
        homeTeam: suggestion.homeTeam,
        awayTeam: suggestion.awayTeam,
        prediction: suggestion.prediction,
        confidence: suggestion.confidence,
        reasoning: suggestion.reasoning,
        odds: suggestion.odds,
        timestamp: suggestion.timestamp || new Date().toISOString()
      }));
    } catch (error) {
      console.error('Error fetching betting suggestions:', error);
      return [];
    }
  }
}

module.exports = BettingAPI; 