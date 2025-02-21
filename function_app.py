import azure.functions as func
import logging
import json
import requests
import uuid
import asyncio
import os

# Load .env file if needed
from dotenv import load_dotenv
load_dotenv()
from azure.cosmos import CosmosClient
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Set up logging config at the start of your file
logging.basicConfig(level=logging.INFO)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Initialize the Semantic Kernel
kernel = Kernel()
base_url = os.getenv("AZURE_SEARCH_BASE_URL", "https://default-url/")
api_version = os.getenv("AZURE_SEARCH_API_VERSION", "2024-07-01")
api_key = os.getenv("AZURE_SEARCH_API_KEY", "default-api-key")
# Add Azure OpenAI chat completion
# Add Azure OpenAI chat completion using environment variables
chat_completion = AzureChatCompletion(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "your-default-api-key"),
    base_url=os.getenv("AZURE_OPENAI_BASE_URL", "https://default-url/")
)
kernel.add_service(chat_completion)

# CosmosDB configuration using environment variables
COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT", "https://default-endpoint/")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY", "your-default-key")
COSMOS_DB_DATABASE_NAME = os.getenv("COSMOS_DB_DATABASE_NAME", "default-database")
COSMOS_DB_CONTAINER_NAME = os.getenv("COSMOS_DB_CONTAINER_NAME", "default-container")

# Initialize the Cosmos client
cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(COSMOS_DB_DATABASE_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER_NAME)

@app.route(route="ai_search_history", methods=["POST"])
def ai_search_history(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        body = req.get_json()
        conversation_id = body.get('conversation_id')
        new_message = body.get('conversation')
        index_to_search = body.get('index_to_search')
        fields = body.get('fields')  # Default value if not provided
        semantic_configuration = body.get('semanticConfiguration')  # Default value if not provided
    except ValueError:
        return func.HttpResponse("Invalid JSON payload", status_code=400)

    if not index_to_search:
        return func.HttpResponse("index_to_search parameter is required.", status_code=400)

    # Retrieve conversation history from Cosmos DB
    existing_conversation = ""
    if conversation_id:
        query = f"SELECT * FROM c WHERE c.conversation_id = '{conversation_id}'"
        try:
            results = container.query_items(query=query, enable_cross_partition_query=True)
            existing_conversation = ";".join([res['conversation'] for res in results])
            logging.info(f"Retrieved conversation history for {conversation_id}: {existing_conversation}")
        except Exception as e:
            logging.error(f"Error retrieving data from Cosmos DB: {e}")
            return func.HttpResponse(f"Error retrieving data from Cosmos DB: {e}", status_code=500)

    # Construct the search query by analyzing the intent
    if existing_conversation:
        search_query = asyncio.run(analyze_intent(existing_conversation, new_message))
    else:
        search_query = new_message

    logging.info(f"Using search query: {search_query}")

    # Save new message to Cosmos DB
    if conversation_id and new_message:
        item = {
            'id': str(uuid.uuid4()),
            'conversation_id': conversation_id,
            'conversation': new_message
        }
        try:
            container.create_item(item)
            logging.info("Data saved to Cosmos DB")
        except Exception as e:
            logging.error(f"Error saving data to Cosmos DB: {e}")
            return func.HttpResponse(f"Error saving data to Cosmos DB: {e}", status_code=500)

    # Modify endpoint to include the specified index_to_search
    azure_search_endpoint = f"{base_url}{index_to_search}/docs/search?api-version={api_version}"

    # Call Azure AI search with a vector search
    search_results = {}
    if search_query:
        search_payload = {
            "search": search_query,
            "count": True,
            "vectorQueries": [
                {
                    "kind": "text",
                    "text": search_query,
                    "fields": fields
                }
            ],
            "queryType": "semantic",
            "semanticConfiguration": semantic_configuration,
            "captions": "extractive",
            "answers": "extractive|count-3"
        }
        
        search_request_headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }

        try:
            response = requests.post(azure_search_endpoint, headers=search_request_headers, data=json.dumps(search_payload))
            response.raise_for_status()
            
            # Log the response content
            logging.info(f"AI Search Response: {response.json()}")  # Or another attribute if the response format needs specific parsing
            
            search_results = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Azure AI search: {e}")
            return func.HttpResponse(f"Error calling Azure AI search: {e}", status_code=500)

    # Return only the "value" field from the results and strip unwanted fields
    raw_results = search_results.get("value", [])
    cleaned_results = []

    for item in raw_results:
        # Remove specific fields
        item.pop("@search.score", None)
        item.pop("@search.rerankerScore", None)
        item.pop("chunk_id", None)
        item.pop("parent_id", None)
        cleaned_results.append(item)

    # Wrap the cleaned results array in an object
    result_object = {
        "results": cleaned_results
    }

    # Return the wrapped results as a JSON response
    return func.HttpResponse(
        json.dumps(result_object),
        status_code=200,
        mimetype="application/json"
    )

async def analyze_intent(chat_history, new_message):
    custom_prompt = f"""
    Given the chat history and new user message, deduce the intent clearly suitable for an AI database search query:

    Chat History: {chat_history}

    New Message: {new_message}

    Simplified Intent:
    """

    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    history = ChatHistory()
    history.add_user_message(custom_prompt)

    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        kernel=kernel,
        settings=execution_settings
    )

    return result.content.strip().lower() if hasattr(result, 'content') else new_message