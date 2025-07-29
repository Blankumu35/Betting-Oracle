# Odds Service

The Odds Service is a component of the Betting Assistant project that handles the retrieval and management of betting odds from various sources. This service is designed to be modular and easily integrated with other services in the project.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd betting-assistant/services/odds-service
   ```

2. **Install Dependencies**
   Ensure you have Node.js installed, then run:
   ```bash
   npm install
   ```

3. **Build the Service**
   Compile the TypeScript files:
   ```bash
   npm run build
   ```

4. **Run the Service**
   Start the service:
   ```bash
   npm start
   ```

## Usage Examples

- **Get Odds**
  To retrieve the latest odds, send a GET request to the `/odds` endpoint.

- **Update Odds**
  To update the odds, send a POST request to the `/odds/update` endpoint with the necessary data.

## API Endpoints

- `GET /odds`: Fetch the current odds.
- `POST /odds/update`: Update the odds with new data.

## Contributing

If you would like to contribute to the Odds Service, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.