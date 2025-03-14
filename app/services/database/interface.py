from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime

class DatabaseInterface(ABC):
    """Abstract interface for database operations."""
    
    # Agent operations
    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID."""
        pass
    
    @abstractmethod
    def create_agent(self, agent_data: Dict[str, Any]) -> str:
        """Create a new agent."""
        pass
    
    @abstractmethod
    def update_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> bool:
        """Update an existing agent."""
        pass
    
    @abstractmethod
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        pass
    
    @abstractmethod
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents."""
        pass
    
    # Conversation operations
    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        pass
    
    @abstractmethod
    def create_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Create a new conversation."""
        pass
    
    @abstractmethod
    def update_conversation(self, conversation_id: str, conversation_data: Dict[str, Any]) -> bool:
        """Update an existing conversation."""
        pass
    
    @abstractmethod
    def list_conversations(self, agent_id: Optional[str] = None, user_id: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """List conversations, optionally filtered by agent_id or user_id."""
        pass
    
    # Message operations
    @abstractmethod
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a message by ID."""
        pass
    
    @abstractmethod
    def create_message(self, message_data: Dict[str, Any]) -> str:
        """Create a new message."""
        pass
    
    @abstractmethod
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        pass

