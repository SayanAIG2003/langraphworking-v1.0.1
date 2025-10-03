# IT Service Management AI Agent
### This project provides a FastAPI backend for an AI-powered agent designed to assist with IT service management tasks. It leverages LangGraph to create a stateful, multi-step agent that can understand user queries, use various tools (retrievers) to find information, and generate comprehensive answers.
## Project Structure
```
.
├── data_dump/                  # Directory for your CSV data files
│   ├── APP-CMDB.csv
│   ├── ChangeTicketsForEmbeddings-v6.csv
│   └── sample_incident_data.csv
├── vector_stores/              # Persisted vector databases will be stored here
├── config/
│   └── bedrock.py              # Initializes AWS Bedrock LLM and embeddings
├── database/
│   └── vector_db.py            # Manages vector store creation and loading
├── retrievers/
│   ├── __init__.py
│   ├── change_retriever.py     # Sets up the change management retriever
│   ├── cmdb_retriever.py       # Sets up the CMDB retriever
│   └── incident_retriever.py   # Sets up the incident retriever
├── graph/
│   ├── __init__.py
│   ├── state.py                # Defines the AgentState TypedDict
│   ├── nodes.py                # Contains all nodes for the LangGraph workflow
│   └── workflow.py             # Builds and compiles the LangGraph
├── tools/
│   └── tool_definitions.py     # Defines the functions that act as agent tools
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```
 
## Setup and Installation
### Prerequisites:
- Python 3.9+
- AWS account with access to Amazon Bedrock models (like Claude 3.5 Sonnet).
- AWS credentials configured in your environment (e.g., via aws configure).
- Install Dependencies:
- Data:
`Place your CSV files (APP-CMDB.csv, ChangeTicketsForEmbeddings-v6.csv, sample_incident_data.csv) inside the data_dump/ directory.
Run the Application:The first time you run the application, it will create and persist the vector stores from your CSV files. This might take a few minutes. Subsequent startups will be much faster as they will load the existing stores.`
 
The API will be available at `http://127.0.0.1:8000`.
### How to Use the API
You can interact with the agent by sending POST requests to the /chat endpoint.
API Endpoint: POST /chat
Request Body:
```bash
{
  "user_input": "What were the recent high priority changes related to the 'global location management system'?",
  "thread_id": "optional-conversation-id-123"
}
```
 
user_input (string, required): The question or command for the agent.
thread_id (string, optional): An identifier for the conversation. If you provide the same thread_id in subsequent requests, the agent will have memory of the past interactions. If omitted, a new one is generated.
Example using curl:
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_input": "hello there"
}'
```


```bash
Successful Response (200 OK):
{
  "response": "Hello! I'm your IT Service Management assistant. How can I help you today?",
  "thread_id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "validation": {
    "relevance_score": 5,
    "completeness_score": 5,
    "clarity_score": 5,
    "overall_score": 5,
    "passed_validation": true,
    "feedback": "The response is a direct and appropriate greeting."
  }
}
```
 
You can view the interactive API documentation provided by FastAPI at http://127.0.0.1:8000/docs.
uvicorn main:app --reload
 
pip install -r requirements.txt