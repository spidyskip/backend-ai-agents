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
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            endpoint_url=settings.S3_ENDPOINT_URL
        )
    
    def get_file(self, folder: str, doc_id: str ) -> Optional[Union[Dict[str, Any], str]]:
        """
        Retrieve a document from S3.
        
        Args:
            doc_id: The ID of the document.
            folder: The folder where the document is stored.
            
        Returns:
            The document as a dictionary if JSON, or as a string if Markdown or plain text, or None if not found.
        """
        try:
            key = f"{folder}/{doc_id}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            
            # Check the content type
            content_type = response.get('ContentType')
            if not content_type:
                # Infer content type based on file extension
                if doc_id.endswith('.json'):
                    content_type = 'application/json'
                elif doc_id.endswith('.md'):
                    content_type = 'text/markdown'
                else:
                    logger.warning(f"Unsupported file extension for document ID {doc_id}")
                    return None
            
            if content_type == 'application/json' or content_type == 'text/markdown':
                # Handle JSON and Markdown content
                content = response['Body'].read().decode('utf-8')
                document = {
                    'folder': folder,
                    'id': doc_id,
                    'content_type': content_type,
                    'content': content,
                    "created_at": response['LastModified'].isoformat(),
                    "updated_at": response['LastModified'].isoformat()
                }
            else:
                logger.warning(f"Unsupported content type: {content_type}")
                return None
            
            return document
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"No document found with ID {doc_id} in folder {folder}")
                return None
            else:
                logger.error(f"Error retrieving document {doc_id} from S3: {str(e)}")
                return None
    
    def add_file(self, doc_id: str, content: Union[Dict[str, Any], str], folder: str ) -> bool:
        """
        Add a file to S3.
        
        Args:
            doc_id: The ID of the document.
            content: The content of the document.
            folder: The folder where the document will be stored.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            if isinstance(content, dict):
                key = f"{folder}/{doc_id}"
                body = json.dumps(content).encode('utf-8')
                content_type = 'application/json'
            else:
                key = f"{folder}/{doc_id}"
                body = content.encode('utf-8')
                content_type = 'text/markdown'
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType=content_type
            )
            return True
        except ClientError as e:
            logger.error(f"Error adding document {doc_id} to S3: {str(e)}")
            return False
    
    def list_files(self, folder: str) -> List[str]:
        """
        List all files in a directory.
        
        Args:
            folder: The folder to list files from.
            
        Returns:
            A list of file names.
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=f"{folder}/")
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])
            return files
        except ClientError as e:
            logger.error(f"Error listing files in directory {folder}: {str(e)}")
            return []
    
    def list_directories(self, folder: str) -> List[str]:
        """
        List all available directories in a folder.
        
        Returns:
            A list of category names
        """
        categories = set()
        
        try:
            # List all objects with the documents/ prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=f"{folder}/", Delimiter='/'):
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        # Extract category from prefix
                        # Format: documents/{category}/
                        category = prefix['Prefix'].split('/')[1]
                        categories.add(category)
        except ClientError as e:
            logger.error(f"Error listing categories from S3: {str(e)}")
        
        return list(categories)
    
    def delete_file(self, folder: str, doc_id: str ) -> bool:
        """
        Delete a file from S3.
        
        Args:
            doc_id: The ID of the document.
            folder: The folder where the document is stored.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            key = f"{folder}/{doc_id}"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            
            # Check the content type
            content_type = response.get('ContentType')
            if not content_type:
                # Infer content type based on file extension
                if doc_id.endswith('.json'):
                    content_type = 'application/json'
                elif doc_id.endswith('.md'):
                    content_type = 'text/markdown'
                else:
                    logger.warning(f"Unsupported file extension for document ID {doc_id}")
                    return None
            
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            logger.error(f"Error deleting document {doc_id} from S3: {str(e)}")
            return False
                