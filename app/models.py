from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base

class AgentConfig(Base):
    __tablename__ = "agent_configs"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    model_name = Column(String, nullable=False)
    tools = Column(JSON, nullable=False)
    categories = Column(JSON, nullable=True)
    keywords = Column(JSON, nullable=True)
    additional_info = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to conversations
    conversations = relationship("Conversation", back_populates="agent")
    
    def __repr__(self):
        return f"<AgentConfig(id='{self.id}', name='{self.name}')>"


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("agent_configs.id"))
    user_id = Column(String, nullable=True)  # Add user_id field
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    agent = relationship("AgentConfig", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id='{self.id}', agent_id='{self.agent_id}', user_id='{self.user_id}')>"


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String, nullable=False)  # 'user', 'assistant', or 'system'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id='{self.id}', role='{self.role}')>"

