from .change_retriever import ChangeRequestRetriever
from .incident_retriever import IncidentRetriever
from .cmdb_retriever import AppCmdbRetriever
import chromadb
from .config import CHROMA_CONFIG
 
# Define global placeholders for the retriever instances.
# These will be populated by the initialization function.
change_retriever = None
cmdb_retriever = None
incident_retriever = None

chroma_client = chromadb.PersistentClient(
    path=CHROMA_CONFIG["persist_directory"]
)
 
def initialize_all_retrievers():
    """
    Explicitly initializes all retrievers and their vector stores.
    This function is called once at application startup.
    """
    # Use a global declaration to modify the module-level variables
    global change_retriever, cmdb_retriever, incident_retriever
    
    # Import necessary clients here to avoid circular dependencies
    from config.bedrock import bedrock_llm, bedrock_embeddings

 
    if not all([bedrock_llm, bedrock_embeddings]):
        print("❌ Cannot initialize retrievers. Bedrock or Chroma clients failed to load.")
        return
 
    try:
        print("🚀 Initializing all retrievers...")
        # This is where the __init__ method of each class is called,
        # which in turn calls initialize_vectorstore() for each retriever.
        print("Initializing ChangeRequestRetriever... 1")
        change_retriever = ChangeRequestRetriever(
            embeddings=bedrock_embeddings, llm=bedrock_llm, chroma_client=chroma_client
        )
        cmdb_retriever = AppCmdbRetriever(
            embeddings=bedrock_embeddings, llm=bedrock_llm, chroma_client=chroma_client
        ).retriever
        incident_retriever = IncidentRetriever(
            embeddings=bedrock_embeddings, llm=bedrock_llm, chroma_client=chroma_client
        )
        print("✅ All retrievers initialized successfully.")
    except Exception as e:
        print(f"❌ A critical error occurred during retriever initialization: {e}")
        # Depending on the application's needs, you might want to exit here
        # raise SystemExit(f"Could not initialize retrievers: {e}")
 
__all__ = ["change_retriever", "cmdb_retriever", "incident_retriever", "initialize_all_retrievers"]
 