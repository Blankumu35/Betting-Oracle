import React from 'react';
import { createRoot, Root } from 'react-dom/client';
import App from './App';
import { ApolloProvider } from '@apollo/client';
import client from './apollo/client';

const container = document.getElementById('root');
if (!container) {
  throw new Error('Failed to find the root element');
}
const root: Root = createRoot(container);

root.render(
  <ApolloProvider client={client}>
    <App />
  </ApolloProvider>
);