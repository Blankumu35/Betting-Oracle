import React, { useState, useRef } from 'react';
import { ApolloProvider } from '@apollo/client';
import client from './apollo/client';
import BettingSuggestions from './components/BettingSuggestions';
import Login from './components/Login';
import Signup from './components/Signup';

const UserDropdown: React.FC<{ onLogout: () => void }> = ({ onLogout }) => {
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  React.useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-3 py-2 rounded-full bg-white shadow hover:bg-gray-50 focus:outline-none"
      >
        {/* Simple user avatar icon */}
        <span className="inline-block w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-white font-bold">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M5.121 17.804A9 9 0 1112 21a8.963 8.963 0 01-6.879-3.196z" />
          </svg>
        </span>
        <svg className="w-4 h-4 ml-1 text-gray-500" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-40 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
          <button
            onClick={onLogout}
            className="block w-full text-left px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-t-lg"
          >
            Log Out
          </button>
        </div>
      )}
    </div>
  );
};

const App: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const isAuthenticated = !!localStorage.getItem('token');

  const toggleAuthMode = () => {
    setIsLogin(!isLogin);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    window.location.reload();
  };

  return (
    <ApolloProvider client={client}>
      <div className="min-h-screen bg-gradient-to-br from-gray-100 to-blue-50">
        {isAuthenticated ? (
          <div className="container mx-auto px-4 py-8">
            <div className="flex justify-between items-center mb-8 border-b border-gray-200 pb-4 shadow-sm bg-white rounded-lg">
              <div className="flex items-center gap-3">
                <span className="inline-block text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600 tracking-tight drop-shadow-sm">
                  <svg className="inline-block w-8 h-8 mr-2 align-middle text-blue-500" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none" />
                    <path d="M8 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Betting Assistant
                </span>
              </div>
              <UserDropdown onLogout={handleLogout} />
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