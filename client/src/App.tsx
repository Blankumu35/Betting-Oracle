import React, { useState } from 'react';
import { ApolloProvider } from '@apollo/client';
import client from './apollo/client';
import BettingSuggestions from './components/BettingSuggestions';
import Login from './components/Login';
import Signup from './components/Signup';

const App: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const isAuthenticated = !!localStorage.getItem('token');

  const toggleAuthMode = () => {
    setIsLogin(!isLogin);
  };

  return (
    <ApolloProvider client={client}>
      <div className="min-h-screen bg-gray-100">
        {isAuthenticated ? (
          <div className="container mx-auto px-4 py-8">
            <div className="flex justify-between items-center mb-8">
              <h1 className="text-3xl font-bold text-gray-800">Betting Assistant</h1>
              <button
                onClick={() => {
                  localStorage.removeItem('token');
                  window.location.reload();
                }}
                className="px-4 py-2 text-sm text-red-600 hover:text-red-800"
              >
                Sign Out
              </button>
            </div>
            <BettingSuggestions />
          </div>
        ) : isLogin ? (
          <Login onToggle={toggleAuthMode} />
        ) : (
          <Signup onToggle={toggleAuthMode} />
        )}
      </div>
    </ApolloProvider>
  );
};

export default App;