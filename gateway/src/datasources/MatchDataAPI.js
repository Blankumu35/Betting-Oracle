const { RESTDataSource } = require('apollo-datasource-rest');

class MatchDataAPI extends RESTDataSource {
  constructor() {
    super();
    this.baseURL = process.env.MATCH_SERVICE_URL || 'http://localhost:4002';
  }

  async getMatchData() {
    try {
      const response = await this.get('/matches');
      return response.map(match => ({
        id: match.id,
        homeTeam: match.homeTeam,
        awayTeam: match.awayTeam,
        date: match.date
      }));
    } catch (error) {
      console.error('Error fetching match data:', error);
      return [];
    }
  }
}

module.exports = MatchDataAPI; 