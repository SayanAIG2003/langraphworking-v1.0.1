import warnings
import chromadb
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langgraph.checkpoint.memory import MemorySaver

# Suppress warnings
warnings.filterwarnings("ignore")

# AWS Bedrock Configuration
AWS_CONFIG = {
    "region_name": "us-west-2",
    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
}

# ChromaDB Configuration
CHROMA_CONFIG = {
    "persist_directory": "./vector_stores"
}

bedrock_llm = None
bedrock_embeddings = None

def initialize_bedrock_components():
    """Initialize Bedrock components"""
    global bedrock_llm, bedrock_embeddings
    
    try:
        # Initialize Bedrock LLM for chat
        bedrock_llm = ChatBedrock(
            model_id=AWS_CONFIG["model_id"],
            region_name=AWS_CONFIG["region_name"],
            model_kwargs={
                "max_tokens": 4000,
                "temperature": 0.1
            }
        )

        # Initialize Bedrock Embeddings
        bedrock_embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=AWS_CONFIG["region_name"]
        )

        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_CONFIG["persist_directory"]
        )
        
        # Initialize memory saver
        memory = MemorySaver()

        print("✅ AWS Bedrock clients initialized successfully!")
        
        return bedrock_embeddings, bedrock_llm, chroma_client, memory

    except Exception as e:
        print(f"❌ Error initializing Bedrock clients: {e}")
        bedrock_llm = None
        bedrock_embeddings = None
        raise e