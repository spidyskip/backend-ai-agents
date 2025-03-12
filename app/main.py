from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import logging
from sqlalchemy.orm import Session
from sqlalchemy.sql import func as sql_func
import uuid

from app.database import get_db, engine, Base
from app.services.agent_manager import AgentManager
from app.schemas import (
    CreateAgentRequest, AgentResponse, ChatRequest, ChatResponse,
    ConversationCreate, ConversationSchema, ConversationWithMessages,
    MessageCreate, MessageSchema
)
from app.models import AgentConfig, Conversation, Message
from app.config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load agents from database on startup
@app.on_event("startup")
async def startup_event():
    db = next(get_db())
    try:
        AgentManager.load_agents_from_db(db)
    finally:
        db.close()

# Route to list all agents
@app.get("/agents", response_model=List[AgentResponse], tags=["Agents"])
async def list_agents(db: Session = Depends(get_db)):
    """List all available agents in the database"""
    agents = AgentManager.list_agents(db)
    return agents

# Route to get a specific agent by ID
@app.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """Get a specific agent by ID"""
    try:
        agent_info = AgentManager.get_agent(agent_id, db)
        # Get the agent config from the database to access model_name and tools
        db_agent = db.query(AgentConfig).filter(AgentConfig.id == agent_id).first()
        if not db_agent:
            raise ValueError(f"No agent found with ID '{agent_id}'.")
            
        return {
            "agent_id": agent_id,
            "name": agent_info["metadata"]["name"],
            "prompt": agent_info["metadata"]["prompt"],
            "model_name": str(db_agent.model_name),
            "tools": db_agent.tools if db_agent.tools is not None else [],
            "categories": agent_info["metadata"].get("categories", []),
            "keywords": agent_info["metadata"].get("keywords", [])
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Route to create a new agent
@app.post("/agent", response_model=AgentResponse, tags=["Agents"])
async def create_agent(request: CreateAgentRequest, db: Session = Depends(get_db)):
    """Create a new agent with specified configuration"""
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
            db,
            categories,
            keywords
        )
        return agent_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Route to handle chat queries and select appropriate agent
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Handle a chat query by selecting the appropriate agent and processing the message.
    If agent_id is provided, that specific agent will be used.
    If not, the system will select the most appropriate agent based on the query.
    """
    try:
        # Ensure thread_id is not None
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Process the chat request through the agent manager
        result = await AgentManager.process_chat(
            request.query,
            request.agent_id,
            thread_id,
            db
        )
        
        # Save the conversation and messages to the database
        # Check if conversation exists
        conversation = db.query(Conversation).filter(Conversation.id == thread_id).first()
        if not conversation:
            conversation = Conversation(
                id=thread_id,
                agent_id=result["agent_id"],
                title=f"Conversation {thread_id[:8]}"
            )
            db.add(conversation)
            db.commit()
        
        # Save user message
        user_message = Message(
            conversation_id=thread_id,
            role="user",
            content=request.query
        )
        db.add(user_message)
        
        # Save assistant message
        assistant_message = Message(
            conversation_id=thread_id,
            role="assistant",
            content=result["response"]
        )
        db.add(assistant_message)
        db.commit()
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Route to create a new conversation
@app.post("/conversations", response_model=ConversationSchema, tags=["Conversations"])
async def create_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db)
):
    """Create a new conversation for a specific agent"""
    try:
        # Check if agent exists
        agent = db.query(AgentConfig).filter(AgentConfig.id == request.agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent with ID {request.agent_id} not found")
        
        # Create new conversation
        conversation = Conversation(
            id=str(uuid.uuid4()),
            agent_id=request.agent_id,
            title=request.title or f"Conversation with {agent.name}"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        return conversation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to get all conversations
@app.get("/conversations", response_model=List[ConversationSchema], tags=["Conversations"])
async def get_conversations(
    agent_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get all conversations, optionally filtered by agent_id"""
    query = db.query(Conversation)
    if agent_id:
        query = query.filter(Conversation.agent_id == agent_id)
    
    conversations = query.order_by(Conversation.updated_at.desc()).offset(skip).limit(limit).all()
    return conversations

# Route to get a specific conversation with its messages
@app.get("/conversations/{conversation_id}", response_model=ConversationWithMessages, tags=["Conversations"])
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific conversation with all its messages"""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    print(conversation)
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
    
    return conversation

# Route to add a message to a conversation
@app.post("/conversations/{conversation_id}/messages", response_model=MessageSchema, tags=["Messages"])
async def add_message(
    conversation_id: str,
    message: MessageCreate,
    db: Session = Depends(get_db)
):
    """Add a new message to a conversation"""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
    
    db_message = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=message.role,
        content=message.content
    )
    db.add(db_message)
    
    # Update conversation timestamp using SQL expression
    db.query(Conversation).filter(Conversation.id == conversation_id).update(
        {"updated_at": sql_func.now()}, 
        synchronize_session=False
    )
    
    db.commit()
    db.refresh(db_message)
    
    # If this is a user message, generate a response
    if message.role == "user" and message.content:
        try:
            # Process chat with the agent
            # Get agent_id as string
            agent_id = None
            if conversation.agent_id is not None:
                agent_id = str(conversation.agent_id)
            
            result = await AgentManager.process_chat(
                message.content,
                agent_id,
                conversation_id,
                db
            )
            
            # Save the assistant response
            assistant_message = Message(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=result["response"]
            )
            db.add(assistant_message)
            db.commit()
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    return db_message

# Route to get all messages for a conversation
@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageSchema], tags=["Messages"])
async def get_messages(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get all messages for a specific conversation"""
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation with ID {conversation_id} not found")
    
    messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
    return messages

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "version": settings.APP_VERSION}

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "documentation": "/docs",
        "endpoints": [
            {"path": "/agents", "method": "GET", "description": "List all available agents"},
            {"path": "/agent", "method": "POST", "description": "Create a new agent"},
            {"path": "/chat", "method": "POST", "description": "Chat with an agent"},
            {"path": "/conversations", "method": "GET", "description": "List all conversations"},
            {"path": "/conversations/{conversation_id}", "method": "GET", "description": "Get a specific conversation"},
            {"path": "/health", "method": "GET", "description": "Health check"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
