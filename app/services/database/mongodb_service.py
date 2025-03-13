from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid

from app.config import settings
from app.services.database.interface import DatabaseInterface

logger = logging.getLogger(__name__)

class MongoDBService(DatabaseInterface):
    """Service for interacting with MongoDB database."""
    
    def __init__(self):
        """Initialize the MongoDB service with credentials from settings."""
        self.client = MongoClient(settings.MONGODB_URI, server_api=ServerApi('1'))
        self.db = self.client[settings.MONGODB_DB_NAME]
        self.agents_collection = self.db["agents"]
        self.conversations_collection = self.db["conversations"]
        self.messages_collection = self.db["messages"]
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create necessary indexes for the collections."""
        try:
            # Agents collection
            self.agents_collection.create_index("id", unique=True)
            
            # Conversations collection
            self.conversations_collection.create_index("id", unique=True)
            self.conversations_collection.create_index("agent_id")
            
            # Messages collection
            self.messages_collection.create_index("id", unique=True)
            self.messages_collection.create_index("conversation_id")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {str(e)}")
    
    # Agent operations
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: The ID of the agent.
            
        Returns:
            The agent document, or None if not found.
        """
        return self.agents_collection.find_one({"id": agent_id})
    
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
        
        # Add timestamps
        now = datetime.utcnow()
        agent_data["created_at"] = now
        agent_data["updated_at"] = now
        
        # Insert the agent
        self.agents_collection.insert_one(agent_data)
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
        # Update timestamp
        agent_data["updated_at"] = datetime.utcnow()
        
        # Update the agent
        result = self.agents_collection.update_one(
            {"id": agent_id},
            {"$set": agent_data}
        )
        
        return result.modified_count > 0
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent.
        
        Args:
            agent_id: The ID of the agent to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        result = self.agents_collection.delete_one({"id": agent_id})
        return result.deleted_count > 0
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all agents.
        
        Returns:
            A list of agent documents.
        """
        return list(self.agents_collection.find())
    
    # Conversation operations
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            The conversation document, or None if not found.
        """
        conversation = self.conversations_collection.find_one({"id": conversation_id})
        if not conversation:
            return None
        
        # Get messages for the conversation
        messages = self.get_conversation_messages(conversation_id)
        
        # Get agent for the conversation
        agent = None
        if "agent_id" in conversation and conversation["agent_id"]:
            agent = self.get_agent(conversation["agent_id"])
        
        # Add messages and agent to the conversation
        conversation["messages"] = messages
        conversation["agent"] = agent
        
        return conversation
    
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
        
        # Add timestamps
        now = datetime.utcnow()
        conversation_data["created_at"] = now
        conversation_data["updated_at"] = now
        
        # Insert the conversation
        self.conversations_collection.insert_one(conversation_data)
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
        # Update timestamp
        conversation_data["updated_at"] = datetime.utcnow()
        
        # Update the conversation
        result = self.conversations_collection.update_one(
            {"id": conversation_id},
            {"$set": conversation_data}
        )
        
        return result.modified_count > 0
    
    def list_conversations(self, agent_id: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List conversations, optionally filtered by agent ID.
        
        Args:
            agent_id: Optional agent ID to filter by.
            skip: Number of documents to skip.
            limit: Maximum number of documents to return.
            
        Returns:
            A list of conversation documents.
        """
        query = {}
        if agent_id:
            query["agent_id"] = agent_id
        
        return list(
            self.conversations_collection.find(query)
            .sort("updated_at", pymongo.DESCENDING)
            .skip(skip)
            .limit(limit)
        )
    
    # Message operations
    
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a message by ID.
        
        Args:
            message_id: The ID of the message.
            
        Returns:
            The message document, or None if not found.
        """
        return self.messages_collection.find_one({"id": message_id})
    
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
        
        # Add timestamp
        message_data["created_at"] = datetime.utcnow()
        
        # Insert the message
        self.messages_collection.insert_one(message_data)
        
        # Update the conversation's updated_at timestamp
        if "conversation_id" in message_data:
            self.conversations_collection.update_one(
                {"id": message_data["conversation_id"]},
                {"$set": {"updated_at": datetime.utcnow()}}
            )
        
        return message_data["id"]
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            A list of message documents.
        """
        return list(
            self.messages_collection.find({"conversation_id": conversation_id})
            .sort("created_at", pymongo.ASCENDING)
        )

