# Betting Assistant API Gateway

The Betting Assistant API Gateway serves as the central point for managing requests to various microservices in the Betting Assistant project. It utilizes GraphQL to provide a flexible and efficient way to interact with the underlying services.

## Project Structure

- **src/index.js**: The entry point for the API Gateway. It sets up the Apollo Server and connects to the GraphQL schema.
- **src/schema.js**: Defines the GraphQL schema, including types and resolvers for handling requests.
- **package.json**: Lists the dependencies and scripts required for the API Gateway, including Apollo Server and GraphQL.

## Getting Started

To get started with the API Gateway, follow these steps:

1. **Install Dependencies**: Run the following command in the `gateway` directory to install the required packages:
   ```
   npm install
   ```

2. **Start the Server**: Use the following command to start the Apollo Server:
   ```
   npm start
   ```

3. **Access the GraphQL Playground**: Once the server is running, you can access the GraphQL Playground at `http://localhost:4000` to test your queries and mutations.

## API Documentation

Refer to the `src/schema.js` file for detailed information about the available queries and mutations, as well as the data types used in the API.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.