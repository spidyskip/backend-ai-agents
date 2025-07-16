# Dynamic Agent Backend with LangGraph

A robust backend system for creating and managing AI agents using LangGraph and FastAPI.

## Features

- **Dynamic Agent Creation**: Create agents with different personalities, tools, and capabilities
- **Automatic Agent Selection**: System automatically selects the most appropriate agent for a given query
- **Stateful Conversations**: Maintain conversation state across multiple interactions
- **Tool Integration**: Agents can use tools like search, calculator, and database lookups
- **Database Persistence**: Agent configurations are stored in a database for persistence

## Project Structure

```
project/
├── app/
│   ├── main.py                 # FastAPI app with main routes
│   ├── models.py               # Database models for agents
│   ├── schemas.py              # Pydantic models for validation
│   ├── database.py             # Database setup
│   ├── tools.py                # Tool definitions
│   └── services/
│       ├── agent_manager.py    # Agent management and selection logic
│       ├── documents/
│       │   ├── interface.py    # Document service interface
│       │   ├── s3_document_service.py  # S3 document service implementation
│       │   ├── local_document_service.py  # Local document service implementation
│       │   └── documents_manager.py  # Document service manager
│       └── storage/
│           └── s3_service.py   # S3 storage service
├── scripts/
│   └── init_db.py              # Script to initialize the database with sample agents
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## API Endpoints

- **GET /agents**: List all available agents
- **POST /agent**: Create a new agent
- **PATCH /agents/{agent_id}**: Update an existing agent
- **POST /chat**: Chat with an agent (automatically selects the best agent if none specified)
- **GET /conversations**: List all conversations
- **POST /conversations**: Create a new conversation
- **GET /conversations/{conversation_id}**: Get a specific conversation with all its messages
- **POST /conversations/{conversation_id}/messages**: Add a new message to a conversation
- **GET /documents**: List all available documents

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
```uvicorn app.main:app --host 0.0.0.0 --port 8000```

3. Access the API documentation at:
```http://localhost:8000/docs```
