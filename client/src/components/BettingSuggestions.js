import React from 'react';
import { useQuery, gql } from '@apollo/client';
import { Card, CardContent, Typography, Grid, CircularProgress, Alert } from '@mui/material';

const GET_BETTING_SUGGESTIONS = gql`
  query GetBettingSuggestions {
    bettingSuggestions {
      matchId
      homeTeam
      awayTeam
      prediction
      confidence
      odds
      reasoning
      timestamp
    }
  }
`;

const BettingSuggestions = () => {
  const { loading, error, data } = useQuery(GET_BETTING_SUGGESTIONS, {
    pollInterval: 30000, // Poll every 30 seconds
  });

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
        <CircularProgress />
      </div>
    );
  }

  if (error) {
    return (
      <Alert severity="error" style={{ margin: '20px' }}>
        Error loading betting suggestions: {error.message}
      </Alert>
    );
  }

  return (
    <div style={{ padding: '20px' }}>
      <Typography variant="h4" gutterBottom>
        Betting Suggestions
      </Typography>
      <Grid container spacing={3}>
        {data.bettingSuggestions.map((suggestion) => (
          <Grid item xs={12} md={6} lg={4} key={suggestion.matchId}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {suggestion.homeTeam} vs {suggestion.awayTeam}
                </Typography>
                <Typography color="textSecondary" gutterBottom>
                  Prediction: {suggestion.prediction}
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Confidence: {suggestion.confidence}%
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Odds: {suggestion.odds}
                </Typography>
                <Typography variant="body2">
                  Reasoning: {suggestion.reasoning}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Last updated: {new Date(suggestion.timestamp).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </div>
  );
};

export default BettingSuggestions; 