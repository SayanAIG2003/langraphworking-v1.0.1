import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Local imports
from config.bedrock import initialize_bedrock_components
from tools.tool_definitions import SmartToolWrapper
from graph.workflow import create_workflow
from retrievers.change_retriever import ChangeRequestRetriever
from retrievers.cmdb_retriever import AppCmdbRetriever  
from retrievers.incident_retriever import IncidentRetriever

# Initialize FastAPI
app = FastAPI(title="IT Service Management Assistant", version="1.0.1")

# Global variables for components
simple_app = None
initialized = False

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    thread_id: str
    retrieved_docs: str
    debug_info: Optional[dict] = None

conversation_history = []

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global simple_app, initialized
    
    try:
        print("🚀 Initializing IT Service Management Assistant...")
        
        # Initialize Bedrock components
        bedrock_embeddings, bedrock_llm, chroma_client, memory = initialize_bedrock_components()
        
        # Initialize retrievers
        print("📚 Initializing retrievers...")
        change_retriever = ChangeRequestRetriever(
            embeddings=bedrock_embeddings,
            llm=bedrock_llm,
            chroma_client=chroma_client
        )

        app_cmdb_retriever = AppCmdbRetriever(
            embeddings=bedrock_embeddings,
            llm=bedrock_llm,
            chroma_client=chroma_client
        )

        incident_retriever = IncidentRetriever(
            embeddings=bedrock_embeddings,
            llm=bedrock_llm,
            chroma_client=chroma_client
        )
        
        # Initialize Smart Tools
        print("🔧 Initializing smart tools...")
        simple_change_tool = SmartToolWrapper(
            retriever=change_retriever,
            llm=bedrock_llm,
            tool_name="change_management",
            data_description="""
            This contains change request data information for each change. Its major information are Change Id, Gear Id, open and close Dates, and other details/description.
            
            Use this for queries about change requests, deployments, maintenance activities, and application changes.
            """
        )

        simple_cmdb_tool = SmartToolWrapper(
            retriever=app_cmdb_retriever,
            llm=bedrock_llm,
            tool_name="cmdb_configuration",
            data_description="""
            This contains application and server configuration data with fields:
    - SERVER_NAME, APPLICATION_NAME
    - GEAR_ID (application identifier)
    - CI_CATEGORY (type of configuration item)
    - DV_INSTALL_STATUS, DV_OPERATIONAL_STATUS
    - DV_SUPPORT_GROUP (team responsible)
    - IP_ADDRESS
   
    Use this for queries about servers, applications, gear IDs, support groups, and infrastructure configuration.
            """
        )

        simple_incident_tool = SmartToolWrapper(
            retriever=incident_retriever,
            llm=bedrock_llm,
            tool_name="incident_management",
            data_description="""
            This contains incident ticket data for each incident, with major information like incident number, gear id, other incident deatils.
   
    Use this for queries about incidents, outages, problems, service issues, and their resolution.
            """
        )
        
        # Create workflow
        print("🔄 Creating workflow...")
        simple_app = create_workflow(memory, bedrock_llm, simple_change_tool, simple_cmdb_tool, simple_incident_tool)
        
        initialized = True
        print("✅ IT Service Management Assistant initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    if not initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {"message": "IT Service Management Assistant is running", "status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the multi-step reasoning agent"""
    
    if not initialized or simple_app is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    thread_id = str(uuid.uuid4())
    conversation_history.append({"role": "user", "content": request.query})

    # Create initial state
    initial_state = {
            "messages": [],
            "user_input": request.query,
            "current_step": "starting",
            "execution_plan": [],
            "current_plan_index": 0,
            "context_data": {},
            "intermediate_results": [],
            "final_response": "",
            "error_recovery_attempts": 0,
            "max_recovery_attempts": 1,
            "thread_id": str(uuid.uuid4()),
            "step_attempt_counts": {},
            "max_step_attempts": 2,
            "total_iterations": 0,
            "max_total_iterations": 10,
            "conversation_history": conversation_history.copy()  # Pass conversation history
        }
    
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10
    }
    
    try:
        print(f"🤔 Processing query: {request.query}")
        result = simple_app.invoke(initial_state, config=config)
        
        debug_info = {
            "steps_executed": len(result.get('intermediate_results', [])),
            "total_iterations": result.get('total_iterations', 0),
            "execution_plan": result.get('execution_plan', [])
        }
        conversation_history.append({"role": "assistant", "content": result['final_response']})
        
        return QueryResponse(
            response=result['final_response'],
            retrieved_docs=str(result.get('retrieved_raw_data', [])),
            thread_id=thread_id,
            debug_info=debug_info
        )
        
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if initialized else "initializing",
        "components": {
            "workflow": simple_app is not None,
            "initialized": initialized
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)