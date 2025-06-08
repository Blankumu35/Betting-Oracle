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
      console.log('Attempting login for:', email);
      const response = await this.post('/api/users/login', { email, password });
      console.log('Login response:', response);
      
      if (!response || !response.id) {
        throw new Error('Invalid response from user service');
      }

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
      throw new Error('Login failed: ' + error.message);
    }
  }

  async signup(email, password) {
    try {
      console.log('Attempting signup for:', email);
      const response = await this.post('/api/users/signup', { email, password });
      console.log('Signup response:', response);

      if (!response || !response.id) {
        throw new Error('Invalid response from user service');
      }

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
      throw new Error('Signup failed: ' + error.message);
    }
  }

  async getUser(email) {
    try {
      console.log('Fetching user:', email);
      const response = await this.get(`/api/users/${email}`);
      console.log('Get user response:', response);
      return response;
    } catch (error) {
      console.error('Error fetching user:', error);
      throw new Error('Failed to fetch user: ' + error.message);
    }
  }
}

module.exports = UserAPI; 