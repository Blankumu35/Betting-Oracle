# User Preference Service

The User Preference Service is a component of the Betting Assistant project that manages user preferences for betting suggestions.  
It is now built using **Java** and **Spring Boot**, providing a robust and scalable solution for handling user-specific data.

## Features

- **User Preference Management**: Allows users to set and update their betting preferences.
- **RESTful API**: Exposes endpoints for interacting with user preferences.
- **Integration**: Works seamlessly with other services in the Betting Assistant ecosystem.

## Getting Started

### Prerequisites

- Java 11 or higher
- Maven

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/betting-assistant.git
   cd betting-assistant/user-pref-service
   ```

2. Build the project:
   ```
   mvn clean install
   ```

3. Run the service:
   ```
   mvn spring-boot:run
   ```

### API Endpoints

- `GET /preferences`: Retrieve user preferences.
- `POST /preferences?user={user}&pref={pref}`: Create or update user preferences.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.