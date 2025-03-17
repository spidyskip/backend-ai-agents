from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import logging
import os
from sqlalchemy.orm import Session
import uuid

from app.database import get_db_session, get_db_service, Base, engine
from app.services.agent_manager import AgentManager
from app.services.documents.documents_manager import get_document_service
from app.schemas import (
    CreateAgentRequest, AgentResponse, ChatRequest, ChatResponse,
    ConversationCreate, ConversationSchema, ConversationWithMessages,
    MessageCreate, MessageSchema, UpdateAgentRequest, DocumentResponse, DocumentCreate,
    UpdateConversationRequest
)
from app.config import settings, DatabaseType

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL), 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if we're running on Vercel
is_vercel = os.environ.get('VERCEL', False)
mock_vercel = os.environ.get('MOCK_VERCEL', False)
if is_vercel:
    logger.info("Running on Vercel")
    if mock_vercel:
        logger.info("Mocking Vercel environment")
else:
    logger.info("Running in development environment")


# Create tables only if using SQLite and not running on Vercel
if settings.DATABASE_TYPE == DatabaseType.SQLITE and not is_vercel:
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

origins = [
    "https://v0-ai-chat-interface-neon.vercel.app",  # URL del tuo frontend
    "http://localhost:3000",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load agents from database on startup
@app.on_event("startup")
async def startup_event():
    # Skip database operations on Vercel
    if is_vercel and mock_vercel:
        logger.info("Running on Vercel without S3, skipping database operations")
        return
    
    try:
        AgentManager.load_agents_from_db()
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {
        "status": "healthy", 
        "version": settings.APP_VERSION,
        "environment": "Vercel" if is_vercel else "Development",
        "storage_type": settings.DATABASE_TYPE
    }

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "environment": "Vercel" if is_vercel else "Development",
        "storage_type": settings.DATABASE_TYPE,
        "endpoints": [
            {"path": "/agents", "method": "GET", "description": "List all available agents"},
            {"path": "/agent", "method": "POST", "description": "Create a new agent"},
            {"path": "/chat", "method": "POST", "description": "Chat with an agent"},
            {"path": "/conversations", "method": "GET", "description": "List all conversations"},
            {"path": "/conversations/{conversation_id}", "method": "GET", "description": "Get a specific conversation"},
            {"path": "/health", "method": "GET", "description": "Health check"}
        ]
    }

# Route to list all agents
@app.get("/agents", response_model=List[AgentResponse], tags=["Agents"])
async def list_agents():
    """List all available agents in the database"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        return []  # Return empty list on Vercel without S3
    
    agents = AgentManager.list_agents()
    return agents

# Route to get a specific agent by ID
@app.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def get_agent(agent_id: str):
    """Get a specific agent by ID"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        raise HTTPException(status_code=404, detail="Agent not found (Vercel environment)")
    
    try:
        agent_info = AgentManager.get_agent(agent_id)
        
        # Get the agent metadata
        metadata = agent_info["metadata"]
        
        # Get the agent from the database to get model_name and tools
        db_service = get_db_service()
        db_agent = db_service.get_agent(agent_id)
        
        if not db_agent:
            raise ValueError(f"No agent found with ID '{agent_id}'.")
            
        return {
            "agent_id": agent_id,
            "name": metadata["name"],
            "prompt": metadata["prompt"],
            "model_name": db_agent["model_name"],
            "tools": db_agent.get("tools", []),
            "categories": metadata.get("categories", []),
            "keywords": metadata.get("keywords", [])
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Route to create a new agent
@app.post("/agent", response_model=AgentResponse, tags=["Agents"])
async def create_agent(request: CreateAgentRequest):
    """Create a new agent with specified configuration"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        raise HTTPException(status_code=400, detail="Cannot create agents in Vercel environment without S3")
    
    try:
        # Generate UUID if agent_id not provided
        agent_id = request.agent_id or str(uuid.uuid4())
        
        # Ensure categories and keywords are lists, not None
        categories = request.categories or []
        keywords = request.keywords or []
        
        agent_info = AgentManager.create_agent(
            agent_id, 
            request.name,
            request.prompt, 
            request.model_name,
            request.tools,
            categories,
            keywords
        )
        return agent_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    

# Route to update agent additional info
@app.patch("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest
):
    """Update the additional information for an agent."""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        raise HTTPException(status_code=400, detail="Cannot update agents in Vercel environment without S3")

    try:
        update_data = request.model_dump(exclude_unset=True)
        updated_agent = AgentManager.update_agent(agent_id, update_data)
        return updated_agent
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Route to handle chat queries and select appropriate agent
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest, 
    background_tasks: BackgroundTasks
):
    """
    Handle a chat query by selecting the appropriate agent and processing the message.
    If agent_id is provided, that specific agent will be used.
    If not, the system will select the most appropriate agent based on the query.
    
    You can optionally provide:
    - user_id and user_info: To include user-specific information in the system prompt
    - additional_prompts: To define additional_prompts like language, units of measurement, etc.
    - include_history: To include previous messages in the conversation
    """
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        # Return a mock response for Vercel environment
        return {
            "response": "I'm running on Vercel and can't access the database. Please use the development environment for full functionality.",
            "agent_id": "mock-agent",
            "agent_name": "Vercel Mock Agent",
            "thread_id": request.thread_id or str(uuid.uuid4()),
            "confidence": 1.0
        }
    
    try:
        # Ensure thread_id is not None
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Process the chat request through the agent manager
        result = await AgentManager.process_chat(
            request.query,
            request.agent_id,
            thread_id,
            request.user_id,
            request.user_info,
            request.additional_prompts,
            request.include_history,
            request.include_documents
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.info(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing chat")

# Route to create a new conversation
@app.post("/conversations", response_model=ConversationSchema, tags=["Conversations"])
async def create_conversation(
    request: ConversationCreate
):
    """Create a new conversation for a specific agent"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        raise HTTPException(status_code=400, detail="Cannot create conversations in Vercel environment without S3")
    
    try:
        # Get database service
        db_service = get_db_service()
        
        # Check if agent exists
        agent = db_service.get_agent(request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent with ID {request.agent_id} not found")
        
        # Create new conversation
        conversation_id = db_service.create_conversation({
            "id": str(uuid.uuid4()),
            "agent_id": request.agent_id,
            "user_id": request.user_id,
            "title": request.title or f"Conversation with {agent['name']}"
        })
        
        # Get the created conversation
        conversation = db_service.get_conversation(conversation_id)
        return conversation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to get all conversations
@app.get("/conversations", response_model=List[ConversationSchema], tags=["Conversations"])
async def get_conversations(
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 10
):
    """Get all conversations, optionally filtered by agent_id or user_id"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        return []  # Return empty list on Vercel without S3
    
    # Get database service
    db_service = get_db_service()
    
    # Get conversations
    conversations = db_service.list_conversations(agent_id, user_id, skip, limit)
    return conversations

# Route to get a specific conversation with its messages
@app.get("/conversations/{conversation_id}", response_model=ConversationWithMessages, tags=["Conversations"])
async def get_conversation(
    conversation_id: str
):
    """Get a specific conversation with all its messages"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        raise HTTPException(status_code=404, detail="Conversation not found (Vercel environment)")
    
    # Get database service
    db_service = get_db_service()
    
    # Get conversation
    conversation = db_service.get_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
    
    return conversation

@app.patch("/conversations/{conversation_id}", response_model=ConversationSchema, tags=["Conversations"])
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest
):
    """Update the title of a specific conversation"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        raise HTTPException(status_code=400, detail="Cannot update conversations in Vercel environment without S3")
    
    # Get database service
    db_service = get_db_service()
    
    # Get conversation
    conversation = db_service.get_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
    
    # Update the conversation title
    try:
        update_data = {"title": request.title}
        success = db_service.update_conversation(conversation_id, update_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update conversation")
        
        updated_conversation = db_service.get_conversation(conversation_id)
        return updated_conversation
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to add a message to a conversation
@app.post("/conversations/{conversation_id}/messages", response_model=MessageSchema, tags=["Messages"])
async def add_message(
    conversation_id: str,
    message: MessageCreate,
    additional_prompts: Optional[Dict[str, Any]] = None,
    include_history: bool = False
):
    """
    Add a new message to a conversation and optionally get a response from the agent.
    
    You can provide:
    - additional_prompts: To define additional_prompts like language, units of measurement, etc.
    - include_history: To include previous messages in the conversation
    """
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        raise HTTPException(status_code=400, detail="Cannot add messages in Vercel environment without S3")
    
    # Get database service
    db_service = get_db_service()
    
    # Check if conversation exists
    conversation = db_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
    
    # Create message
    message_id = db_service.create_message({
        "id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
        "role": message.role,
        "content": message.content
    })
    
    # Get the created message
    db_message = db_service.get_message(message_id)
    
    # If this is a user message, generate a response
    if message.role == "user" and message.content:
        try:
            # Process chat with the agent
            # Get agent_id as string
            agent_id = None
            if "agent_id" in conversation and conversation["agent_id"]:
                agent_id = str(conversation["agent_id"])
            
            # Get user_id if available
            user_id = None
            if "user_id" in conversation and conversation["user_id"]:
                user_id = str(conversation["user_id"])
            
            result = await AgentManager.process_chat(
                message.content,
                agent_id,
                conversation_id,
                user_id,
                None,  # No user_info for existing conversations
                additional_prompts,
                include_history
            )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    return db_message

# Route to get all messages for a conversation
@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageSchema], tags=["Messages"])
async def get_messages(
    conversation_id: str
):
    """Get all messages for a specific conversation"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        return []  # Return empty list on Vercel without S3
    
    # Get database service
    db_service = get_db_service()
    
    # Check if conversation exists
    conversation = db_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
    
    # Get messages
    messages = db_service.get_conversation_messages(conversation_id)
    return messages

# Route get list documents
@app.get("/documents", response_model=List[DocumentResponse], tags=["Documents"])
async def list_documents():
    """List all available documents in the database"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        return []
    document_service = get_document_service()
    documents = document_service.list_documents()
    return documents

@app.post("/documents", response_model=DocumentResponse, tags=["Documents"])
async def add_document(
    category: str,
    document: DocumentCreate
):
    """Add a new document to the specified category"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        raise HTTPException(status_code=400, detail="Cannot add documents in Vercel environment without S3")
    
    document_service = get_document_service()
    
    try:
        doc_data = document.dict()
        doc_id = document_service.add_document(category, doc_data)
        created_document = document_service.get_document(category, doc_id)
        return created_document
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{category_id}/{doc_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document(
    category: str,
    doc_id: str
):
    """Get a specific document by ID"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        return []
    document_service = get_document_service()
    try:
        document = document_service.get_document(category, [doc_id])
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found in category {category}")
        return document
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/categories", tags=["Documents"])
async def list_categories():
    """List all available document categories"""
    # Check if we're on Vercel without S3
    if is_vercel and mock_vercel:
        return []  # Return empty list on Vercel without S3
    
    document_service = get_document_service()
    categories = document_service.list_categories()

    return {"categories": categories}