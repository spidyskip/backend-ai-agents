from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import uuid
from datetime import datetime

# Base schemas
class AgentBase(BaseModel):
    name: str
    prompt: str
    model_name: str
    tools: List[str]
    categories: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

# Create agent request schema
class CreateAgentRequest(AgentBase):
    agent_id: Optional[str] = None
    
    @validator('agent_id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())

# Agent response schema
class AgentResponse(AgentBase):
    agent_id: str
    
    class Config:
        orm_mode = True

# Chat request schema
class ChatRequest(BaseModel):
    query: str
    agent_id: Optional[str] = None
    thread_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

# Chat response schema
class ChatResponse(BaseModel):
    response: str
    agent_id: str
    agent_name: str
    thread_id: str
    confidence: float

# Create conversation schema
class ConversationCreate(BaseModel):
    agent_id: str
    title: Optional[str] = None

# Conversation schema
class ConversationSchema(BaseModel):
    id: str
    agent_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

# Message base schema
class MessageBase(BaseModel):
    role: str
    content: str

# Create message schema
class MessageCreate(MessageBase):
    pass

# Message schema
class MessageSchema(MessageBase):
    id: str
    conversation_id: str
    created_at: datetime
    
    class Config:
        orm_mode = True

# Agent schema for nested relationships
class AgentNestedSchema(BaseModel):
    id: str
    name: str
    prompt: str
    model_name: str
    tools: List[str]
    categories: List[str]
    keywords: List[str]
    
    class Config:
        orm_mode = True

# Conversation with messages schema
class ConversationWithMessages(BaseModel):
    id: str
    agent_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: List[MessageSchema] = []
    agent: Optional[AgentNestedSchema] = None
    
    class Config:
        orm_mode = True
