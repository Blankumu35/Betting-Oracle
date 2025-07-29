import express from 'express';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware setup
app.use(express.json());

// Define routes here
app.get('/users', (req, res) => {
    res.send('User service is running');
});

// Start the server
app.listen(PORT, () => {
    console.log(`User service is running on port ${PORT}`);
});