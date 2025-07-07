# Betting Assistant Project

The Betting Assistant project is a modern application that provides users with football match predictions and betting suggestions, powered by machine learning. The system is designed for extensibility and integrates a React frontend, a GraphQL gateway, and a Python-based ML backend.

## Project Structure

The project is organized into the following main directories:

- **client/**: The frontend application built with React, Tailwind CSS, and Apollo Client. It displays football predictions and betting suggestions in a user-friendly interface.
- **gateway/**: The GraphQL API Gateway implemented with Node.js and Apollo Server. It acts as a single entry point for the frontend, aggregating data from the ML backend and other sources.
- **ml/**: The machine learning backend, built with Python and FastAPI. It serves football match predictions (outcome and over/under 2.5 goals) via a REST API, and is integrated with the gateway.
- **recommendation-engine/**: Contains the football stats agent and scraping logic for collecting and analyzing football data (fixtures, team form, H2H, etc.).

> **Note:** Previous services for trend analysis, match data, and user preferences have been removed for simplicity and focus.

## Technologies Used

- **Frontend**: React, Tailwind CSS, Apollo Client
- **API Gateway**: Node.js, Apollo Server
- **ML Backend**: Python, FastAPI, scikit-learn, joblib
- **Data Collection**: Playwright, BeautifulSoup (for scraping football stats)
- **Containerization**: Docker, Docker Compose

## Machine Learning Pipeline

The ML backend predicts:
- **Match Outcome**: Home Win, Draw, or Away Win
- **Over/Under 2.5 Goals**: Whether the match will have more or fewer than 2.5 goals

### Features Used for Prediction
- Recent team form (wins, draws, losses)
- Team strength proxies (goal difference, rolling averages)
- Elo-like features (from historical data)
- Match context (league importance, month, weekend)
- Simple odds proxies (if available)
- Rolling win rates and volatility

> **Note:** Head-to-head (H2H) stats are collected and available, but not currently used as ML features by default.

### How It Works
1. The stats agent scrapes and analyzes upcoming fixtures.
2. Features are extracted for each fixture.
3. The trained ML model predicts outcomes and over/under goals, returning probabilities and confidence scores.
4. Predictions are served via FastAPI and routed through the GraphQL gateway to the frontend.

## Getting Started

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd betting-assistant
   ```

2. **Set up the services:**
   - Install dependencies for each service:
     - `client/`: `npm install`
     - `gateway/`: `npm install`
     - `ml/`: `pip install -r requirements.txt`
   - (Optional) Train or retrain the ML models in `ml/` if needed.

3. **Run the services:**
   - Start the ML backend:
     ```
     cd ml
     uvicorn predict_fixtures:app --host 0.0.0.0 --port 8001
     ```
   - Start the GraphQL gateway:
     ```
     cd gateway
     npm start
     ```
   - Start the frontend:
     ```
     cd client
     npm start
     ```

4. **Access the application:**
   - Open your browser at `http://localhost:3000` to use the Betting Assistant.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.