import React from 'react';
import { useQuery, gql } from '@apollo/client';
import { Card, CardContent, Typography, Grid, CircularProgress, Alert } from '@mui/material';

const GET_BETTING_SUGGESTIONS = gql`
  query GetBettingSuggestions {
    bettingSuggestions {
      fixture
      predicted_outcome
      confidence
      over_under
      over_under_confidence
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
        {data.bettingSuggestions.map((suggestion, idx) => (
          <Grid item xs={12} md={6} lg={4} key={idx}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {suggestion.fixture}
                </Typography>
                <Typography color="textSecondary" gutterBottom>
                  Prediction: {suggestion.predicted_outcome}
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Confidence: {(suggestion.confidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Over/Under 2.5: {suggestion.over_under}
                </Typography>
                <Typography variant="body2" gutterBottom>
                  O/U Confidence: {suggestion.over_under_confidence !== undefined ? (suggestion.over_under_confidence * 100).toFixed(1) + '%' : '-'}
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