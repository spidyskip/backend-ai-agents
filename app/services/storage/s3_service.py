import boto3
import json
import logging
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Any, Union

from app.config import settings

logger = logging.getLogger(__name__)

class S3Service:
    """Service for interacting with AWS S3 bucket."""
    
    def __init__(self):
        """Initialize the S3 service with credentials from settings."""
        self.bucket_name = settings.S3_BUCKET_NAME
        self.s3_client = boto3.client(
            's3',
            region_name=settings.S3_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
    
    def get_agent_prompt(self, agent_id: str) -> Optional[str]:
        """
        Retrieve an agent prompt from S3.
        
        Args:
            agent_id: The ID of the agent.
            
        Returns:
            The agent prompt as a string, or None if not found.
        """
        try:
            key = f"agents/{agent_id}/prompt.txt"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            prompt = response['Body'].read().decode('utf-8')
            return prompt
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"No prompt found for agent {agent_id} in S3")
                return None
            else:
                logger.error(f"Error retrieving prompt for agent {agent_id} from S3: {str(e)}")
                return None
    
    def save_agent_prompt(self, agent_id: str, prompt: str) -> bool:
        """
        Save an agent prompt to S3.
        
        Args:
            agent_id: The ID of the agent.
            prompt: The prompt to save.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            key = f"agents/{agent_id}/prompt.txt"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=prompt.encode('utf-8'),
                ContentType='text/plain'
            )
            return True
        except ClientError as e:
            logger.error(f"Error saving prompt for agent {agent_id} to S3: {str(e)}")
            return False
    
    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an agent configuration from S3.
        
        Args:
            agent_id: The ID of the agent.
            
        Returns:
            The agent configuration as a dictionary, or None if not found.
        """
        try:
            key = f"agents/{agent_id}/config.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            config = json.loads(response['Body'].read().decode('utf-8'))
            return config
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"No configuration found for agent {agent_id} in S3")
                return None
            else:
                logger.error(f"Error retrieving configuration for agent {agent_id} from S3: {str(e)}")
                return None
    
    def save_agent_config(self, agent_id: str, config: Dict[str, Any]) -> bool:
        """
        Save an agent configuration to S3.
        
        Args:
            agent_id: The ID of the agent.
            config: The configuration to save.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            key = f"agents/{agent_id}/config.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(config).encode('utf-8'),
                ContentType='application/json'
            )
            return True
        except ClientError as e:
            logger.error(f"Error saving configuration for agent {agent_id} to S3: {str(e)}")
            return False
    
    def list_agents(self) -> List[str]:
        """
        List all agents in the S3 bucket.
        
        Returns:
            A list of agent IDs.
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix="agents/",
                Delimiter="/"
            )
            
            agent_ids = []
            if 'CommonPrefixes' in response:
                for prefix in response['CommonPrefixes']:
                    # Extract agent ID from prefix (format: "agents/{agent_id}/")
                    prefix_path = prefix['Prefix']
                    agent_id = prefix_path.split('/')[1]
                    agent_ids.append(agent_id)
            
            return agent_ids
        except ClientError as e:
            logger.error(f"Error listing agents in S3: {str(e)}")
            return []

