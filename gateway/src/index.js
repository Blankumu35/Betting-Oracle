const { ApolloServer } = require('apollo-server');
const { typeDefs, resolvers } = require('./schema');
const BettingAPI = require('./datasources/BettingAPI');
const UserAPI = require('./datasources/UserAPI');
const { getUser } = require('./auth');

const server = new ApolloServer({
  typeDefs,
  resolvers,
  dataSources: () => ({
    bettingAPI: new BettingAPI(),
   // userAPI: new UserAPI()
  }),
  context: ({ req }) => {
    // Get the user token from the headers
    const token = req.headers.authorization || '';
    
    // Try to retrieve a user with the token
    const user = getUser(token.replace('Bearer ', ''));
    
    // Add the user to the context
    return { user };
  },
  cors: {
    origin: 'http://localhost:3000', // Frontend URL
    credentials: true // Allow credentials
  }
});

const PORT = process.env.PORT || 4000;

server.listen(PORT).then(({ url }) => {
  console.log(`🚀  Server ready at ${url}`);
});