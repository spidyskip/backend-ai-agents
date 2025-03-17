import boto3
import json
import logging
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from app.config import settings
from app.services.storage.s3_service import S3Service
from app.services.database.interface import DatabaseInterface

logger = logging.getLogger(__name__)

class S3DatabaseService(DatabaseInterface):
    """Service for interacting with AWS S3 bucket as a database."""
    
    def __init__(self):
        """Initialize the S3 database service with credentials from settings."""
        self.s3_service = S3Service()
    
    # Agent operations
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID."""
        return self.s3_service.get_file("agents", f"{agent_id}.json")
    
    def create_agent(self, agent_data: Dict[str, Any]) -> str:
        """Create a new agent."""
        agent_id = agent_data.get('id', str(uuid.uuid4()))
        agent_data['id'] = agent_id
        agent_data['created_at'] = datetime.utcnow().isoformat()
        agent_data['updated_at'] = datetime.utcnow().isoformat()
        success = self.s3_service.add_file(f"{agent_id}.json", agent_data, "agents")
        if success:
            return agent_id
        else:
            raise ValueError("Failed to create agent")
    
    def update_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> bool:
        """Update an existing agent."""
        agent_data['id'] = agent_id
        agent_data['updated_at'] = datetime.utcnow().isoformat()
        return self.s3_service.add_file(f"{agent_id}.json", agent_data, "agents")
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        return self.s3_service.delete_file("agents", f"{agent_id}.json")
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents."""
        agent_files = self.s3_service.list_files("agents")
        agents = []
        for file_key in agent_files:
            if file_key.endswith('.json'):
                agent_id = file_key.split('/')[-1].replace('.json', '')
                agent = self.get_agent(agent_id)
                if agent:
                    agents.append(agent)
        return agents
    
    # Conversation operations
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        return self.s3_service.get_file("conversations", f"{conversation_id}.json")
    
    def create_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Create a new conversation."""
        conversation_id = conversation_data.get('id', str(uuid.uuid4()))
        conversation_data['id'] = conversation_id
        conversation_data['created_at'] = datetime.utcnow().isoformat()
        conversation_data['updated_at'] = datetime.utcnow().isoformat()
        success = self.s3_service.add_file(f"{conversation_id}.json", conversation_data, "conversations")
        if success:
            return conversation_id
        else:
            raise ValueError("Failed to create conversation")
    
    def update_conversation(self, conversation_id: str, conversation_data: Dict[str, Any]) -> bool:
        """Update an existing conversation."""
        conversation_data['id'] = conversation_id
        conversation_data['updated_at'] = datetime.utcnow().isoformat()
        return self.s3_service.add_file(f"{conversation_id}.json", conversation_data, "conversations")
    
    def list_conversations(self, agent_id: Optional[str] = None, user_id: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """List conversations, optionally filtered by agent_id or user_id."""
        conversation_files = self.s3_service.list_files("conversations")
        conversations = []
        for file_key in conversation_files:
            if file_key.endswith('.json'):
                conversation_id = file_key.split('/')[-1].replace('.json', '')
                conversation = self.get_conversation(conversation_id)
                if conversation:
                    if agent_id and conversation.get('agent_id') != agent_id:
                        continue
                    if user_id and conversation.get('user_id') != user_id:
                        continue
                    conversations.append(conversation)
        return conversations[skip:skip + limit]
    
    # Message operations
    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a message by ID."""
        return self.s3_service.get_file("messages", f"{message_id}.json")
    
    def create_message(self, message_data: Dict[str, Any]) -> str:
        """Create a new message."""
        message_id = message_data.get('id', str(uuid.uuid4()))
        message_data['id'] = message_id
        message_data['created_at'] = datetime.utcnow().isoformat()
        message_data['updated_at'] = datetime.utcnow().isoformat()
        success = self.s3_service.add_file(f"{message_id}.json", message_data, "messages")
        if success:
            return message_id
        else:
            raise ValueError("Failed to create message")
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        message_files = self.s3_service.list_files("messages")
        messages = []
        for file_key in message_files:
            if file_key.endswith('.json'):
                message_id = file_key.split('/')[-1].replace('.json', '')
                message = self.get_message(message_id)
                if message and message.get('conversation_id') == conversation_id:
                    messages.append(message)
        return messages