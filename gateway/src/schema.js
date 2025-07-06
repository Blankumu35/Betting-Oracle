const { gql } = require('apollo-server');

const typeDefs = gql`
  type Query {
    bettingSuggestions: [BettingSuggestion]
    matchData: [Match]
    trends: [Trend]
    me: User
  }

  type Mutation {
    login(email: String!, password: String!): AuthPayload
    signup(email: String!, password: String!): AuthPayload
  }

  type User {
    id: ID!
    email: String!
  }

  type AuthPayload {
    token: String!
    user: User!
  }

  type BettingSuggestion {
    fixture: String!
    predicted_outcome: String!
    confidence: Float!
    over_under: String
    over_under_confidence: Float
  }

  type Match {
    id: ID!
    homeTeam: String!
    awayTeam: String!
    date: String!
  }

  type Trend {
    id: ID!
    description: String!
    value: Float!
  }
`;

const resolvers = {
  Query: {
    bettingSuggestions: async (_, __, { dataSources }) => {
      return dataSources.bettingAPI.getSuggestions();
    },
    matchData: async (_, __, { dataSources }) => {
      return dataSources.matchDataAPI.getMatchData();
    },
    trends: async (_, __, { dataSources }) => {
      return dataSources.trendAPI.getTrends();
    },
    me: (_, __, { user }) => user
  },
  Mutation: {
    login: async (_, { email, password }, { dataSources }) => {
      const user = await dataSources.userAPI.login(email, password);
      if (!user) {
        throw new Error('Invalid credentials');
      }
      return user;
    },
    signup: async (_, { email, password }, { dataSources }) => {
      const user = await dataSources.userAPI.signup(email, password);
      if (!user) {
        throw new Error('Email already in use');
      }
      return user;
    }
  }
};

module.exports = { typeDefs, resolvers };