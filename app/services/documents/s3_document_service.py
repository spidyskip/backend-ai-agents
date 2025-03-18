import boto3
import json
import logging
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime

from app.config import settings
from app.services.documents.interface import DocumentInterface
from app.services.storage.s3_service import S3Service

logger = logging.getLogger(__name__)

class S3DocumentService(DocumentInterface):
    """Service for interacting with documents stored in AWS S3 bucket."""
    
    def __init__(self):
        """Initialize the S3 document service with credentials from settings."""
        self.s3_service = S3Service()    
    
    def get_document(self, category: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by category and ID.
        
        Args:
            category: The document category
            doc_id: The document ID
            
        Returns:
            The document data as a dictionary, or None if not found
        """
        document = self.s3_service.get_file(f"docs/{category}", doc_id)
        if document['content_type'] == 'application/json':
                content = json.loads(document['content'])
                content['id'] = document['id']
                content['category'] = category
                return content
        elif document['content_type'] == 'text/markdown':
            return {
                'id': document['id'],
                'category': category,
                'title': document['id'],
                'content': document['content'],
                'content_length': document.get('content_length', None),
                'created_at': document["created_at"],
                'updated_at': document["created_at"]
            }
        return None
    
    def list_documents(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all documents, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            A list of document metadata
        """
        documents = []
        
        try:
            # If category is provided, list only documents in that category
            if category:
                prefix = f"docs/{category}"
            else:
                prefix = "docs"
            
            files = self.s3_service.list_files(prefix)
            
            for file_key in files:
                # Skip directory objects
                if file_key.endswith('/'):
                    continue
                
                # Only process JSON and Markdown files
                if not (file_key.endswith('.json') or file_key.endswith('.md')):
                    continue
                
                try:
                    # Extract category and document ID from key
                    # Format: docs/{category}/{doc_id}.json or docs/{category}/{doc_id}.md
                    parts = file_key.split('/')
                    category = parts[-2]
                    doc_id = parts[-1].split('.')[0]
                    doc_ext = parts[-1].split('.')[-1]
                    document = self.get_document(f"{category}", f"{doc_id}.{doc_ext}")
                    if document:
                       
                        documents.append(document)
                        
                except Exception as e:
                    logger.error(f"Error processing document {file_key}: {str(e)}")
        except ClientError as e:
            logger.error(f"Error listing documents from S3: {str(e)}")
        
        return documents
    
    def add_document(self, category: str, doc_data: Union[Dict[str, Any], str]) -> str:
        """
        Add a new document to the specified category.
        
        Args:
            category: The document category
            doc_data: The document data (either a dictionary for JSON or a string for Markdown)
            
        Returns:
            The ID of the created document
        """
        try:
            # Generate a document ID if not provided
            doc_id = str(uuid.uuid4())
            
            # Determine the content type and key
            if isinstance(doc_data, dict):
                content_type = 'application/json'
                key = f"docs/{category}/{doc_id}.json"
                doc_data['id'] = doc_id
                doc_data['category'] = category
                doc_data['created_at'] = datetime.utcnow().isoformat()
                doc_data['updated_at'] = datetime.utcnow().isoformat()
            elif isinstance(doc_data, str):
                content_type = 'text/markdown'
                key = f"docs/{category}/{doc_id}.md"
            else:
                raise ValueError("Unsupported document type")
            
            # Save to S3
            self.s3_service.add_file(doc_id, doc_data, f"docs/{category}")
            
            return doc_id
        except ClientError as e:
            logger.error(f"Error adding document to S3: {str(e)}")
            raise ValueError(f"Failed to add document: {str(e)}")
    
    def update_document(self, category: str, doc_id: str, doc_data: Dict[str, Any]) -> bool:
        """
        Update an existing document.
        
        Args:
            category: The document category
            doc_id: The document ID
            doc_data: The updated document data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if document exists
            existing_doc = self.get_document(category, doc_id)
            if not existing_doc:
                return False
            
            # Update metadata
            doc_data['id'] = doc_id
            doc_data['category'] = category
            doc_data['created_at'] = existing_doc.get('created_at', datetime.utcnow().isoformat())
            doc_data['updated_at'] = datetime.utcnow().isoformat()
            
            # Save to S3
            self.s3_service.add_file(doc_id, doc_data, f"docs/{category}")
            
            return True
        except ClientError as e:
            logger.error(f"Error updating document {doc_id} in S3: {str(e)}")
            return False
    
    def delete_document(self, category: str, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            category: The document category
            doc_id: The document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_service.delete_file(f"docs/{category}", doc_id)
            return True
        except ClientError as e:
            logger.error(f"Error deleting document {doc_id} from S3: {str(e)}")
            return False
    
    def list_categories(self) -> List[str]:
        """
        List all available document categories.
        
        Returns:
            A list of category names
        """
        return self.s3_service.list_directories("docs")
    
    def get_documents_by_ids(self, category: str, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get multiple documents by their IDs within a category.
        
        Args:
            category: The document category
            doc_ids: List of document IDs to retrieve
            
        Returns:
            A list of document data
        """
        documents = []
        
        for doc_id in doc_ids:
            document = self.get_document(category, doc_id)
            if document:
                # Add category and ID to the document
                document['category'] = category
                document['id'] = doc_id
                documents.append(document)
        
        return documents