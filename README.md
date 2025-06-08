# Betting Assistant Project

The Betting Assistant project is a comprehensive application designed to provide users with betting suggestions, match data, and trend analysis. It consists of multiple services, each responsible for a specific aspect of the application, and is built using a variety of modern technologies.

## Project Structure

The project is organized into several directories, each containing a specific service:

- **client/**: The frontend of the application built with React, Tailwind CSS, and Apollo Client.
- **gateway/**: The GraphQL API Gateway implemented with Node.js and Apollo Server.
- **recommendation-engine/**: A recommendation engine using Python and Flask to provide betting suggestions based on machine learning algorithms.
- **match-data-service/**: A service written in Go that fetches and serves match statistics.
- **trend-service/**: A trend analysis service implemented in Rust using Actix-web.
- **user-pref-service/**: A user preference service built with Scala and Akka HTTP.

## Technologies Used

- **Frontend**: React, Tailwind CSS, Apollo Client
- **API Gateway**: Node.js, Apollo Server
- **Recommendation Engine**: Python, Flask
- **Match Data Service**: Go
- **Trend Analysis Service**: Rust, Actix-web
- **User Preference Service**: Scala, Akka HTTP
- **Containerization**: Docker, Docker Compose

## Getting Started

To get started with the Betting Assistant project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd betting-assistant
   ```

2. Set up the services:
   - Navigate to each service directory and follow the instructions in their respective README files to install dependencies and run the services.

3. Use Docker Compose to orchestrate the services:
   ```
   docker-compose up
   ```

4. Access the client application in your web browser at `http://localhost:3000`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.