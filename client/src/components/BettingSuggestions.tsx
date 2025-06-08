import React from 'react';
import { useQuery, gql } from '@apollo/client';

const GET_BETTING_SUGGESTIONS = gql`
  query GetBettingSuggestions {
    bettingSuggestions {
      id
      suggestion
      odds
    }
  }
`;

const BettingSuggestions: React.FC = () => {
  const { loading, error, data } = useQuery(GET_BETTING_SUGGESTIONS);

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold">Betting Suggestions</h2>
      <ul className="mt-4">
        {data.bettingSuggestions.map((suggestion: { id: string; suggestion: string; odds: number }) => (
          <li key={suggestion.id} className="border-b py-2">
            <span className="font-semibold">{suggestion.suggestion}</span> - Odds: {suggestion.odds}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default BettingSuggestions;