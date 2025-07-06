import React, { useEffect, useState } from 'react';

interface Prediction {
  fixture: string;
  predicted_outcome: string;
  confidence: number;
  over_under?: string;
  over_under_confidence?: number;
}

const BettingSuggestions: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetch('http://localhost:8001/api/predictions')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setPredictions(data);
          setError(null);
        } else {
          setError(data.error || 'Unknown error');
        }
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="p-8 text-center">Loading predictions...</div>;
  if (error) return <div className="p-8 text-red-600">Error: {error}</div>;

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Predictions</h2>
      <table className="min-w-full table-auto border">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-4 py-2 text-left">Fixture</th>
            <th className="px-4 py-2 text-left">Predicted Outcome</th>
            <th className="px-4 py-2 text-left">Confidence</th>
            <th className="px-4 py-2 text-left">Over/Under 2.5</th>
            <th className="px-4 py-2 text-left">O/U Confidence</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((p, idx) => (
            <tr key={idx} className="border-t">
              <td className="px-4 py-2">{p.fixture}</td>
              <td className="px-4 py-2">{p.predicted_outcome}</td>
              <td className="px-4 py-2">{(p.confidence * 100).toFixed(1)}%</td>
              <td className="px-4 py-2">{p.over_under || '-'}</td>
              <td className="px-4 py-2">{p.over_under_confidence !== undefined ? (p.over_under_confidence * 100).toFixed(1) + '%' : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default BettingSuggestions;