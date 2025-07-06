const { RESTDataSource } = require('apollo-datasource-rest');

class BettingAPI extends RESTDataSource {
  constructor() {
    super();
    this.baseURL = 'http://localhost:8001'; // FastAPI ML service
  }

  async getSuggestions() {
    try {
      const response = await this.get('/api/predictions');
      return response.map(suggestion => ({
        fixture: suggestion.fixture,
        predicted_outcome: suggestion.predicted_outcome,
        confidence: suggestion.confidence,
        over_under: suggestion.over_under,
        over_under_confidence: suggestion.over_under_confidence
      }));
    } catch (error) {
      console.error('Error fetching betting suggestions:', error);
      return [];
    }
  }
}

module.exports = BettingAPI; 