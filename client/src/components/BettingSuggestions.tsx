import React from 'react';
import { useQuery, gql } from '@apollo/client';

const GET_BETTING_SUGGESTIONS = gql`
  query GetBettingSuggestions {
    bettingSuggestions {
      id
      matchId
      homeTeam
      awayTeam
      prediction
      confidence
      reasoning
      odds
      timestamp
    }
  }
`;

interface BettingSuggestion {
  id: string;
  matchId: string;
  homeTeam: string;
  awayTeam: string;
  prediction: string;
  confidence: number;
  reasoning: string;
  odds: number;
  timestamp: string;
}

const BettingSuggestions: React.FC = () => {
  const { loading, error, data } = useQuery(GET_BETTING_SUGGESTIONS);

  if (loading) return (
    <div className="flex justify-center items-center p-8">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
    </div>
  );
  
  if (error) return (
    <div className="p-4 bg-red-100 text-red-700 rounded-lg">
      Error: {error.message}
    </div>
  );

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Betting Suggestions</h2>
      <div className="space-y-6">
        {data.bettingSuggestions.map((suggestion: BettingSuggestion) => (
          <div key={suggestion.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start mb-3">
              <div>
                <h3 className="text-lg font-semibold text-gray-800">
                  {suggestion.homeTeam} vs {suggestion.awayTeam}
                </h3>
                <p className="text-sm text-gray-500">
                  {new Date(suggestion.timestamp).toLocaleDateString()}
                </p>
              </div>
              <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                Odds: {suggestion.odds.toFixed(2)}
              </div>
            </div>
            
            <div className="mb-3">
              <div className="flex items-center gap-2">
                <span className="font-medium text-gray-700">Prediction:</span>
                <span className="text-green-600 font-semibold">{suggestion.prediction}</span>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className="font-medium text-gray-700">Confidence:</span>
                <div className="w-32 bg-gray-200 rounded-full h-2.5">
                  <div 
                    className="bg-blue-600 h-2.5 rounded-full" 
                    style={{ width: `${suggestion.confidence * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm text-gray-600">
                  {(suggestion.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            
            <p className="text-gray-600 text-sm italic">
              "{suggestion.reasoning}"
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BettingSuggestions;