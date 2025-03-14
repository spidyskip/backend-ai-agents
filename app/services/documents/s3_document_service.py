import boto3
import json
import logging
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime

from app.config import settings
from app.services.documents.interface import DocumentInterface

logger = logging.getLogger(__name__)

class S3DocumentService(DocumentInterface):
    """Service for interacting with documents stored in AWS S3 bucket."""
    
    def __init__(self):
        """Initialize the S3 document service with credentials from settings."""
        self.bucket_name = settings.S3_BUCKET_NAME
        self.s3_client = boto3.client(
            's3',
            region_name=settings.S3_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
    
    def get_document(self, category: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by category and ID.
        
        Args:
            category: The document category
            doc_id: The document ID
            
        Returns:
            The document data as a dictionary, or None if not found
        """
        try:
            key = f"documents/{category}/{doc_id}.json"
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            document = json.loads(response['Body'].read().decode('utf-8'))
            return document
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"No document found with ID {doc_id} in category {category}")
                return None
            else:
                logger.error(f"Error retrieving document {doc_id} from S3: {str(e)}")
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
                prefix = f"documents/{category}/"
            else:
                prefix = "documents/"
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Skip directory objects
                        if obj['Key'].endswith('/'):
                            continue
                        
                        # Only process JSON files
                        if not obj['Key'].endswith('.json'):
                            continue
                        
                        try:
                            # Extract category and document ID from key
                            # Format: documents/{category}/{doc_id}.json
                            parts = obj['Key'].split('/')
                            if len(parts) >= 3:
                                doc_category = parts[1]
                                doc_id = parts[2].replace('.json', '')
                                
                                # Get the document
                                document = self.get_document(doc_category, doc_id)
                                if document:
                                    # Add category and ID to the document
                                    document['category'] = doc_category
                                    document['id'] = doc_id
                                    documents.append(document)
                        except Exception as e:
                            logger.error(f"Error processing document {obj['Key']}: {str(e)}")
        except ClientError as e:
            logger.error(f"Error listing documents from S3: {str(e)}")
        
        return documents
    
    def add_document(self, category: str, doc_data: Dict[str, Any]) -> str:
        """
        Add a new document to the specified category.
        
        Args:
            category: The document category
            doc_data: The document data
            
        Returns:
            The ID of the created document
        """
        try:
            # Generate a document ID if not provided
            doc_id = doc_data.get('id', str(uuid.uuid4()))
            
            # Add metadata
            doc_data['id'] = doc_id
            doc_data['category'] = category
            doc_data['created_at'] = datetime.utcnow().isoformat()
            doc_data['updated_at'] = datetime.utcnow().isoformat()
            
            # Save to S3
            key = f"documents/{category}/{doc_id}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(doc_data).encode('utf-8'),
                ContentType='application/json'
            )
            
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
            key = f"documents/{category}/{doc_id}.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(doc_data).encode('utf-8'),
                ContentType='application/json'
            )
            
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
            key = f"documents/{category}/{doc_id}.json"
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=key
            )
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
        categories = set()
        
        try:
            # List all objects with the documents/ prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix="documents/", Delimiter='/'):
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        # Extract category from prefix
                        # Format: documents/{category}/
                        category = prefix['Prefix'].split('/')[1]
                        categories.add(category)
        except ClientError as e:
            logger.error(f"Error listing categories from S3: {str(e)}")
        
        return list(categories)
    
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

