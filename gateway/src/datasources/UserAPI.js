const { RESTDataSource } = require('apollo-datasource-rest');
const jwt = require('jsonwebtoken');
const { JWT_SECRET } = require('../auth');

class UserAPI extends RESTDataSource {
  constructor() {
    super();
    this.baseURL = process.env.USER_SERVICE_URL || 'http://localhost:4004';
  }

  async login(email, password) {
    try {
      const response = await this.post('/login', { email, password });
      const token = jwt.sign({ id: response.id, email: response.email }, JWT_SECRET);
      return {
        token,
        user: {
          id: response.id,
          email: response.email
        }
      };
    } catch (error) {
      console.error('Error during login:', error);
      return null;
    }
  }

  async signup(email, password) {
    try {
      const response = await this.post('/signup', { email, password });
      const token = jwt.sign({ id: response.id, email: response.email }, JWT_SECRET);
      return {
        token,
        user: {
          id: response.id,
          email: response.email
        }
      };
    } catch (error) {
      console.error('Error during signup:', error);
      return null;
    }
  }
}

module.exports = UserAPI; 