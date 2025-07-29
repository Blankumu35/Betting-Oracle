import express from 'express';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware setup
app.use(express.json());

// Routes setup
app.get('/bets', (req, res) => {
    res.send('List of bets');
});

// Start the server
app.listen(PORT, () => {
    console.log(`Bet service is running on port ${PORT}`);
});