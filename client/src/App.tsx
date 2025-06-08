import React from 'react';
import { ApolloProvider } from '@apollo/client';
import client from './apollo/client';
import BettingSuggestions from './components/BettingSuggestions';

const App: React.FC = () => {
  return (
    <ApolloProvider client={client}>
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <h1 className="text-3xl font-bold mb-4">Betting Assistant</h1>
        <BettingSuggestions />
      </div>
    </ApolloProvider>
  );
};

export default App;