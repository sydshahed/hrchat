import os
from pathlib import Path
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from config import ASSET_PATH, get_logger

from azure.ai.inference.prompts import PromptTemplate
from azure.search.documents.models import VectorizedQuery

# Initialize logging and tracing objects
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Environment variables
search_service_name = os.environ["SEARCH_SERVICE_NAME"]
resource_group_name = os.environ["RESOURCE_GROUP_NAME"]
search_index_name = os.environ["AISEARCH_INDEX_NAME"]
search_endpoint = f"https://{search_service_name}.search.windows.net"
search_api_key = os.environ["SEARCH_API_KEY"]

print(search_endpoint)
# Initialize the Search Index Client
search_index_client = SearchIndexClient(
    endpoint=search_endpoint, credential=AzureKeyCredential(search_api_key)
)

# Initialize AI Project Client
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# Create a vector embeddings client that will be used to generate vector embeddings
chat = project.inference.get_chat_completions_client()
embeddings = project.inference.get_embeddings_client()

# Create a search index client using the search connection
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index_name,
    credential=AzureKeyCredential(search_api_key),
)


def list_available_indexes():
    """List all indexes in the Azure Cognitive Search service."""
    logger.info("Fetching list of available indexes...")
    indexes = search_index_client.list_indexes()
    index_names = [index.name for index in indexes]
    logger.info(f"Available indexes: {index_names}")
    return index_names


@tracer.start_as_current_span(name="get_product_documents")
def get_product_documents(messages: list, context: dict = None) -> dict:
    if context is None:
        context = {}

    overrides = context.get("overrides", {})
    top = overrides.get("top", 5)

    # Generate a search query from the chat messages
    intent_prompt = PromptTemplate.from_prompty(Path(ASSET_PATH) / "intent_mapping.prompty")

    intent_mapping_response = chat.complete(
        model=os.environ["INTENT_MAPPING_MODEL"],
        messages=intent_prompt.create_messages(conversation=messages),
        **intent_prompt.parameters,
    )

    search_query = intent_mapping_response.choices[0].message.content
    logger.debug(f"🧠 Intent mapping: {search_query}")

    # Generate a vector representation of the search query
    embedding = embeddings.embed(model=os.environ["EMBEDDINGS_MODEL"], input=search_query)
    search_vector = embedding.data[0].embedding
    logger.debug(f"🔢 Embedding data: {embedding.data}")

    # Search the index for products matching the search query
    vector_query = VectorizedQuery(vector=search_vector, k_nearest_neighbors=top, fields="contentVector")

    search_results = search_client.search(
        search_text=search_query, vector_queries=[vector_query], select=["id", "content", "filepath", "title", "url"]
    )

    documents = [
        {
            "id": result["id"],
            "content": result["content"],
            "filepath": result["filepath"],
            "title": result["title"],
            "url": result["url"],
        }
        for result in search_results
    ]

    # Add results to the provided context
    if "thoughts" not in context:
        context["thoughts"] = []

    # Add thoughts and documents to the context object so it can be returned to the caller
    context["thoughts"].append(
        {
            "title": "Generated search query",
            "description": search_query,
        }
    )

    if "grounding_data" not in context:
        context["grounding_data"] = []
    context["grounding_data"].append(documents)

    logger.debug(f"📄 {len(documents)} documents retrieved: {documents}")
    return documents


if __name__ == "__main__":
    import logging
    import argparse

    # Set logging level to debug when running this module directly
    logger.setLevel(logging.DEBUG)

    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Query to use to search product",
        default="I need a new tent for 4 people, what would you recommend?",
    )

    args = parser.parse_args()
    query = args.query

    try:
        indexes = list_available_indexes()
        if search_index_name not in indexes:
            logger.error(f"Index {search_index_name} does not exist. Please create it.")
        else:
            result = get_product_documents(messages=[{"role": "user", "content": query}])
            print(result)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
