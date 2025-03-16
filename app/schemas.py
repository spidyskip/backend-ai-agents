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
    additional_query: Optional[Dict[str, Any]] = None
    document_refs: Optional[Dict[str, List[str]]] = None  # Added document references

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
        from_attributes = True

# Update agent request schema
class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    model_name: Optional[str] = None
    tools: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    additional_query: Optional[Dict[str, Any]] = None
    document_refs: Optional[Dict[str, List[str]]] = None

# Update agent document refs request schema
class UpdateAgentDocumentRefsRequest(BaseModel):
    document_refs: Dict[str, List[str]]

# Chat request schema
class ChatRequest(BaseModel):
    query: str
    agent_id: Optional[str] = None
    thread_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
    additional_prompts: Optional[Dict[str, Any]] = None
    include_history: bool = False
    include_documents: bool = True  # Whether to include document content in context

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
    user_id: Optional[str] = None
    title: Optional[str] = None

# Conversation schema
class ConversationSchema(BaseModel):
    id: str
    agent_id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class UpdateConversationRequest(BaseModel):
    title: str

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
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Agent schema for nested relationships
class AgentNestedSchema(BaseModel):
    id: str
    name: str
    prompt: str
    model_name: str
    tools: List[str]
    categories: List[str]
    keywords: List[str]
    additional_query: Optional[Dict[str, Any]] = None
    document_refs: Optional[Dict[str, List[str]]] = None
    
    class Config:
        from_attributes = True

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
        from_attributes = True

# Document base schema
class DocumentBase(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

# Create document request schema
class CreateDocumentRequest(DocumentBase):
    id: Optional[str] = None
    
    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())

# Update document request schema
class UpdateDocumentRequest(DocumentBase):
    pass

# Document response schema
class DocumentResponse(DocumentBase):
    id: str
    category: str
    created_at: str
    updated_at: str

class DocumentCreate(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None