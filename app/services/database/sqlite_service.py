import logging
from typing import Dict, List, Optional, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.sql import func as sql_func
import uuid
from datetime import datetime

from app.models import AgentConfig, Conversation, Message
from app.services.database.interface import DatabaseInterface

logger = logging.getLogger(__name__)

class SQLiteService(DatabaseInterface):
    """Service for interacting with SQLite database."""
    
    def __init__(self, db_session: Session):
        """Initialize the SQLite service with a database session."""
        self.db = db_session
    
    # Agent operations
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: The ID of the agent.
            
        Returns:
            The agent as a dictionary, or None if not found.
        """
        agent = self.db.query(AgentConfig).filter(AgentConfig.id == agent_id).first()
        if not agent:
            return None
        
        return {
            "id": agent.id,
            "name": agent.name,
            "prompt": agent.prompt,
            "model_name": agent.model_name,
            "tools": list(agent.tools) if agent.tools is not None and hasattr(agent.tools, '__iter__') else [],
            "categories": list(agent.categories) if agent.categories is not None and hasattr(agent.categories, '__iter__') else [],
            "keywords": list(agent.keywords) if agent.keywords is not None and hasattr(agent.keywords, '__iter__') else [],
            "additional_query": agent.additional_query if hasattr(agent, 'additional_query') else {},
            "document_refs": agent.document_refs if hasattr(agent, 'document_refs') else {},
            "created_at": agent.created_at,
            "updated_at": agent.updated_at
        }
    
    def create_agent(self, agent_data: Dict[str, Any]) -> str:
        """
        Create a new agent.
        
        Args:
            agent_data: The agent data to insert.
            
        Returns:
            The ID of the created agent.
        """
        # Ensure agent has an ID
        if "id" not in agent_data:
            agent_data["id"] = str(uuid.uuid4())
        
        # Create the agent
        agent = AgentConfig(
            id=agent_data["id"],
            name=agent_data["name"],
            prompt=agent_data["prompt"],
            model_name=agent_data["model_name"],
            tools=agent_data.get("tools", []),
            categories=agent_data.get("categories", []),
            keywords=agent_data.get("keywords", []),
            additional_query=agent_data.get("additional_query", {}),
            document_refs=agent_data.get("document_refs", {})
        )
        
        self.db.add(agent)
        self.db.commit()
        
        return agent_data["id"]
    
    def update_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> bool:
        """
        Update an existing agent.
        
        Args:
            agent_id: The ID of the agent to update.
            agent_data: The new agent data.
            
        Returns:
            True if successful, False otherwise.
        """
        agent = self.db.query(AgentConfig).filter(AgentConfig.id == agent_id).first()
        if not agent:
            return False
        
        # Update fields
        if "name" in agent_data:
            agent.name = agent_data["name"]
        if "prompt" in agent_data:
            agent.prompt = agent_data["prompt"]
        if "model_name" in agent_data:
            agent.model_name = agent_data["model_name"]
        if "tools" in agent_data:
            agent.tools = agent_data["tools"]
        if "categories" in agent_data:
            agent.categories = agent_data["categories"]
        if "keywords" in agent_data:
            agent.keywords = agent_data["keywords"]
        if "additional_query" in agent_data:
            agent.additional_query = agent_data["additional_query"]
        if "document_refs" in agent_data:
            agent.document_refs = agent_data["document_refs"]
    
        self.db.commit()
        return True
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.
        
        Args:
            agent_id: The ID of the agent to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        agent = self.db.query(AgentConfig).filter(AgentConfig.id == agent_id).first()
        if not agent:
            return False
        
        self.db.delete(agent)
        self.db.commit()
        return True
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all agents.
        
        Returns:
            A list of agent dictionaries.
        """
        agents = self.db.query(AgentConfig).all()
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "prompt": agent.prompt,
                "model_name": agent.model_name,
                "tools": list(agent.tools) if agent.tools is not None and hasattr(agent.tools, '__iter__') else [],
                "categories": list(agent.categories) if agent.categories is not None and hasattr(agent.categories, '__iter__') else [],
                "keywords": list(agent.keywords) if agent.keywords is not None and hasattr(agent.keywords, '__iter__') else [],
                "additional_query": agent.additional_query if hasattr(agent, 'additional_query') else {},
                "document_refs": agent.document_refs if hasattr(agent, 'document_refs') else {},
                "created_at": agent.created_at,
                "updated_at": agent.updated_at
            }
            for agent in agents
        ]
    
    # Conversation operations
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            The conversation as a dictionary, or None if not found.
        """
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            return None
        
        # Get messages for the conversation
        messages = self.get_conversation_messages(conversation_id)
        
        # Get agent for the conversation
        agent = None
        if conversation.agent_id:
            agent = self.get_agent(conversation.agent_id)
        
        return {
            "id": conversation.id,
            "agent_id": conversation.agent_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": messages,
            "agent": agent
        }
    
    def create_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """
        Create a new conversation.
        
        Args:
            conversation_data: The conversation data to insert.
        
        Returns:
            The ID of the created conversation.
        """
        # Ensure conversation has an ID
        if "id" not in conversation_data:
            conversation_data["id"] = str(uuid.uuid4())
        
        # Create the conversation
        conversation = Conversation(
            id=conversation_data["id"],
            agent_id=conversation_data.get("agent_id"),
            user_id=conversation_data.get("user_id"),  # Add user_id
            title=conversation_data.get("title")
        )
        
        self.db.add(conversation)
        self.db.commit()
        
        return conversation_data["id"]
    
    def update_conversation(self, conversation_id: str, conversation_data: Dict[str, Any]) -> bool:
        """
        Update an existing conversation.
        
        Args:
            conversation_id: The ID of the conversation to update.
            conversation_data: The new conversation data.
        
        Returns:
            True if successful, False otherwise.
        """
        conversation = self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conversation:
            return False
        
        # Update fields
        if "agent_id" in conversation_data:
            conversation.agent_id = conversation_data["agent_id"]
        if "user_id" in conversation_data:  # Add user_id
            conversation.user_id = conversation_data["user_id"]
        if "title" in conversation_data:
            conversation.title = conversation_data["title"]
        
        # Update timestamp
        conversation.updated_at = datetime.utcnow()
        
        self.db.commit()
        return True
    
    def list_conversations(self, agent_id: Optional[str] = None, user_id: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List conversations, optionally filtered by agent_id or user_id.
        
        Args:
            agent_id: Optional agent ID to filter by.
            user_id: Optional user ID to filter by.
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            
        Returns:
            A list of conversation dictionaries.
        """
        query = self.db.query(Conversation)
        if agent_id:
            query = query.filter(Conversation.agent_id == agent_id)
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        
        conversations = query.order_by(Conversation.updated_at.desc()).offset(skip).limit(limit).all()
        
        return [
            {
                "id": conversation.id,
                "agent_id": conversation.agent_id,
                "user_id": conversation.user_id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at
            }
            for conversation in conversations
        ]
    
    # Message operations
    
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a message by ID.
        
        Args:
            message_id: The ID of the message.
            
        Returns:
            The message as a dictionary, or None if not found.
        """
        message = self.db.query(Message).filter(Message.id == message_id).first()
        if not message:
            return None
        
        return {
            "id": message.id,
            "conversation_id": message.conversation_id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at
        }
    
    def create_message(self, message_data: Dict[str, Any]) -> str:
        """
        Create a new message.
        
        Args:
            message_data: The message data to insert.
            
        Returns:
            The ID of the created message.
        """
        # Ensure message has an ID
        if "id" not in message_data:
            message_data["id"] = str(uuid.uuid4())
        
        # Create the message
        message = Message(
            id=message_data["id"],
            conversation_id=message_data["conversation_id"],
            role=message_data["role"],
            content=message_data["content"]
        )
        
        self.db.add(message)
        
        # Update conversation timestamp
        self.db.query(Conversation).filter(Conversation.id == message_data["conversation_id"]).update(
            {"updated_at": sql_func.now()}, 
            synchronize_session=False
        )
        
        self.db.commit()
        
        return message_data["id"]
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            A list of message dictionaries.
        """
        messages = self.db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
        
        return [
            {
                "id": message.id,
                "conversation_id": message.conversation_id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at
            }
            for message in messages
        ]

