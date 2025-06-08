# Match Data Service

The Match Data Service is a crucial component of the Betting Assistant project. It is responsible for fetching and serving match statistics that are essential for making informed betting decisions.

## Overview

This service is implemented in Go and serves as an API that provides match data to other components of the Betting Assistant application, such as the recommendation engine and the API gateway.

## Features

- Fetches live and historical match data.
- Provides endpoints for querying match statistics.
- Integrates with other services in the Betting Assistant ecosystem.

## Getting Started

### Prerequisites

- Go 1.16 or later
- Any necessary dependencies specified in the `go.mod` file

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd betting-assistant/match-data-service
   ```

2. Install dependencies:
   ```
   go mod tidy
   ```

### Running the Service

To run the Match Data Service, execute the following command:

```
go run main.go
```

The service will start and listen for incoming requests.

### API Endpoints

- `/matches`: Get a list of matches.
- `/matches/{id}`: Get details for a specific match.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.