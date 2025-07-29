# User Service Documentation

## Overview

The User Service is a crucial component of the Betting Assistant project, responsible for managing user-related functionalities such as registration, authentication, and profile management.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/betting-assistant.git
   cd betting-assistant/services/user-service
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Run the Service**
   ```bash
   npm start
   ```

## Usage Examples

- **Register a New User**
  - Endpoint: `POST /api/users/register`
  - Request Body:
    ```json
    {
      "username": "exampleUser",
      "password": "examplePassword"
    }
    ```

- **Authenticate User**
  - Endpoint: `POST /api/users/login`
  - Request Body:
    ```json
    {
      "username": "exampleUser",
      "password": "examplePassword"
    }
    ```

## API Reference

- **GET /api/users/:id**
  - Retrieves user information by user ID.

- **PUT /api/users/:id**
  - Updates user information.

- **DELETE /api/users/:id**
  - Deletes a user account.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.