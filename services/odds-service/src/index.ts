import express from 'express';
import { json } from 'body-parser';
import { oddsRouter } from './routes/odds';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(json());
app.use('/odds', oddsRouter);

app.listen(PORT, () => {
    console.log(`Odds service is running on port ${PORT}`);
});