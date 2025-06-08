const { RESTDataSource } = require('apollo-datasource-rest');

class TrendAPI extends RESTDataSource {
  constructor() {
    super();
    this.baseURL = process.env.TREND_SERVICE_URL || 'http://localhost:4003';
  }

  async getTrends() {
    try {
      const response = await this.get('/trends');
      return response.map(trend => ({
        id: trend.id,
        description: trend.description,
        value: trend.value
      }));
    } catch (error) {
      console.error('Error fetching trends:', error);
      return [];
    }
  }
}

module.exports = TrendAPI; 