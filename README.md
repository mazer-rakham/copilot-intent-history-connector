# Cosmos DB History POC Function App

This repository contains a serverless application built on Azure Functions to handle search history and inquiries into Cosmos DB. This README provides guidance on setting up and understanding the components involved in this project.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Functionality](#functionality)
5. [Environment Variables](#environment-variables)
6. [Usage](#usage)
7. [Logging](#logging)
8. [Contributing](#contributing)

## Overview

This project aims to create a flexible consumption function app on Azure that interfaces with Azure Cosmos DB. The application analyzes query requests, searches the Cosmos DB, and returns structured results based on the input.

## Prerequisites

- Azure account with required permissions
- Azure CLI installed
- Python 3.7+ for local development and testing
- [Visual Studio Code](https://code.visualstudio.com/) with Azure Functions Extension and Python Extension
- Semantic Kernel SDK
- OpenAI API access for using deployment and chat completion

## Architecture

The application consists of several key components:
- **Azure Function App**: Handles incoming requests, processes them, and interacts with Cosmos DB.
- **Azure Cosmos DB**: Stores conversation history and other necessary data.
- **Azure Storage Account**: Required for Azure Functions to store triggers and logs.
- **Semantic Kernel SDK**: Used for processing and managing AI-driven operations.

## Functionality

### Key Features
- **Search Query Handling**: Accepts search queries from clients, prepares them using Semantic Kernel, and searches Cosmos DB.
- **Dynamic Configuration**: Uses environment variables for configuration, improving security and flexibility.
- **Structured Results**: Converts output data into a JSON object format for consumption by downstream systems.

### Request Structure
- Endpoint: `/api/ai_search_history`
- Method: `POST`
- Sample Payload:
  ```json
  {
    "conversation_id": "sample-id",
    "conversation": "Check presence of case number",
    "index_to_search": "case-detail-index",
    "fields": "case_vector",
    "semanticConfiguration": "case-detail-semantic-configuration"
  }
## Response Structure

The application returns results wrapped in a JSON object structure. The structure is as follows:

    ```json
    {
      "results": [
        // Array of result objects
      ]
    }

## Environment Variables
Certain environment variables need to be set in your environment to ensure the application functions correctly. These include:

- **AZURE_SEARCH_BASE_URL:** The base URL for the Azure Search service endpoints.
- **AZURE_SEARCH_API_VERSION:** The version of the Azure Search API to use.
- **AZURE_OPENAI_DEPLOYMENT_NAME:** The name of the OpenAI deployment utilized in your app.
- **AZURE_OPENAI_API_KEY:** The API key for accessing the OpenAI service.
- **AZURE_OPENAI_BASE_URL:** The base URL for OpenAI API endpoints.
- **COSMOS_DB_ENDPOINT:** The endpoint URL for your Azure Cosmos DB.
- **COSMOS_DB_KEY:** The key for accessing your Cosmos DB instance.
- **COSMOS_DB_DATABASE_NAME:** The name of the Cosmos DB database in use.
- **COSMOS_DB_CONTAINER_NAME:** The name of the Cosmos DB container storing your data.
- **AZURE_SEARCH_API_KEY:** The API key used for accessing Azure Search.

Each of these should be configured in your deployment environment or within a .env file if you're working locally.
# AI Search History API Custom Connector

This custom connector allows you to integrate the AI Search History API with Power Automate. The API processes conversation_id, conversation, index_to_search, fields, and semanticConfiguration from the request body to return AI search results.

## Uploading the Swagger File to Create a Custom Connector

To create a custom connector in Power Automate using the provided Swagger file, follow these steps:

1. **Sign in to Power Automate**:
   - Go to [Power Automate](https://flow.microsoft.com) and sign in with your account.

2. **Navigate to Custom Connectors**:
   - In the left-hand navigation pane, select **Data** > **Custom connectors**.

3. **Create a New Custom Connector**:
   - Click on **+ New custom connector** and select **Import an OpenAPI file**.

4. **Upload the Swagger File**:
   - Provide a name for your custom connector.
   - Click on **Import** and upload the `swagger.json` file from your local machine.

5. **Configure the Connector**:
   - Review the information imported from the Swagger file and make any necessary adjustments.
   - Ensure that the host, base path, and schemes are correctly set.
   - Click **Continue** to proceed through the configuration steps.

6. **Set Up Security**:
   - Configure the authentication type required by your API (e.g., API key, OAuth 2.0, etc.).
   - Provide the necessary details for authentication.

7. **Define the Actions and Triggers**:
   - Review the actions and triggers defined in the Swagger file.
   - Make any necessary adjustments to the parameters and responses.

8. **Test the Connector**:
   - Test the connector by providing sample data and verifying the responses.
   - Ensure that the connector works as expected.

9. **Create and Publish**:
   - Once testing is complete, click **Create connector**.
   - Publish the connector to make it available for use in your Power Automate flows.

## What the Swagger File Does

The provided Swagger file defines the AI Search History API with the following endpoints:

- **GET /ai_search_history**:
  - Retrieves the details of the request received by the AI Search History endpoint.
  - Response includes method, headers, params, and body.

- **POST /ai_search_history**:
  - Processes the request payload containing conversation_id, conversation, index_to_search, fields, and semanticConfiguration.
  - Returns AI search results based on conversation history and inferred intent.
  - Possible responses include:
    - `200`: Successfully returns AI search results.
    - `400`: Invalid request payload or missing parameters.
    - `500`: Error processing request.

The API is designed to handle AI search history operations, providing detailed request information and processing search requests to return relevant results based on the provided parameters.

By following the steps above, you can create a custom connector in Power Automate that leverages the AI Search History API, enabling you to automate workflows that involve AI-driven search capabilities.

## Usage
After deploying the function, use tools like Postman or curl to interact with the function endpoint for testing. Ensure that the environment variables are properly configured for the Azure Function to access necessary resources.

## Logging
The application logs critical operations:

- Search queries being utilized.
- Responses from Azure AI Search.

These logs can be accessed through Azure Monitor or any connected logging service.

## Contributing
- Fork the repository.
- Create a feature branch.
- Commit changes to your branch.
- Open a pull request against the main branch.

Contributions are welcome! Please respect the coding guidelines and share an issue if you encounter any problems.