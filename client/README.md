# Betting Assistant Client

This directory contains the client application for the Betting Assistant project, built with React, Tailwind CSS, and Apollo Client for GraphQL integration.

## Project Structure

- **src/**: Contains the source code for the React application.
  - **App.tsx**: The main component that sets up the Apollo Provider and renders the application layout.
  - **index.tsx**: The entry point for the React application, rendering the App component into the DOM.
  - **apollo/**: Contains the Apollo Client configuration.
    - **client.ts**: Initializes the Apollo Client for GraphQL requests.
  - **components/**: Contains reusable React components.
    - **BettingSuggestions.tsx**: Fetches and displays betting suggestions using Apollo Client.
  - **styles/**: Contains styles for the application.
    - **tailwind.css**: Tailwind CSS styles for the application.

- **public/**: Contains static files for the application.
  - **index.html**: The HTML template for the React application.

- **package.json**: Lists the dependencies and scripts for the React application.

- **tailwind.config.js**: Configures Tailwind CSS, specifying paths to template files.

- **tsconfig.json**: TypeScript configuration file specifying compiler options.

## Getting Started

To get started with the client application, follow these steps:

1. **Install Dependencies**: Run `npm install` in the `client` directory to install the required dependencies.

2. **Run the Application**: Use `npm start` to start the development server. The application will be available at `http://localhost:3000`.

3. **Build for Production**: Use `npm run build` to create an optimized build of the application for production.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.