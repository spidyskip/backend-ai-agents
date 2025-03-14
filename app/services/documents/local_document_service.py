import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime
import shutil

from app.config import settings
from app.services.documents.interface import DocumentInterface

logger = logging.getLogger(__name__)

class LocalDocumentService(DocumentInterface):
    """Service for interacting with documents stored in the local filesystem."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the local document service.
        
        Args:
            base_dir: Base directory for document storage. Defaults to 'documents' in the current directory.
        """
        self.base_dir = base_dir or os.path.join(os.getcwd(), 'docs')
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _get_category_path(self, category: str) -> str:
        """Get the path to a category directory."""
        category_path = os.path.join(self.base_dir, category)
        os.makedirs(category_path, exist_ok=True)
        return category_path
    
    def _get_document_path(self, category: str, doc_id: str) -> str:
        """Get the path to a document file."""
        return os.path.join(self._get_category_path(category), f"{doc_id}.json")
    
    def get_document(self, category: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by category and ID.
        
        Args:
            category: The document category
            doc_id: The document ID
            
        Returns:
            The document data as a dictionary, or None if not found
        """
        doc_path = self._get_document_path(category, doc_id)
        
        if not os.path.exists(doc_path):
            logger.warning(f"No document found with ID {doc_id} in category {category}")
            return None
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                document = json.load(f)
                return document
        except Exception as e:
            logger.error(f"Error reading document {doc_id}: {str(e)}")
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
        
        # If category is provided, list only documents in that category
        if category:
            categories = [category]
        else:
            categories = self.list_categories()
        
        for cat in categories:
            category_path = self._get_category_path(cat)
            
            # Skip if category directory doesn't exist
            if not os.path.exists(category_path):
                continue
            
            # List all JSON files in the category directory
            for filename in os.listdir(category_path):
                if filename.endswith('.json'):
                    doc_id = filename.replace('.json', '')
                    
                    # Get the document
                    document = self.get_document(cat, doc_id)
                    if document:
                        # Add category and ID to the document
                        document['category'] = cat
                        document['id'] = doc_id
                        documents.append(document)
        
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
        # Generate a document ID if not provided
        doc_id = doc_data.get('id', str(uuid.uuid4()))
        
        # Add metadata
        doc_data['id'] = doc_id
        doc_data['category'] = category
        doc_data['created_at'] = datetime.utcnow().isoformat()
        doc_data['updated_at'] = datetime.utcnow().isoformat()
        
        # Save to file
        doc_path = self._get_document_path(category, doc_id)
        
        try:
            with open(doc_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)
            
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
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
        doc_path = self._get_document_path(category, doc_id)
        
        # Check if document exists
        if not os.path.exists(doc_path):
            return False
        
        try:
            # Read existing document to get created_at timestamp
            existing_doc = self.get_document(category, doc_id)
            
            # Update metadata
            doc_data['id'] = doc_id
            doc_data['category'] = category
            doc_data['created_at'] = existing_doc.get('created_at', datetime.utcnow().isoformat())
            doc_data['updated_at'] = datetime.utcnow().isoformat()
            
            # Save to file
            with open(doc_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {str(e)}")
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
        doc_path = self._get_document_path(category, doc_id)
        
        # Check if document exists
        if not os.path.exists(doc_path):
            return False
        
        try:
            os.remove(doc_path)
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def list_categories(self) -> List[str]:
        """
        List all available document categories.
        
        Returns:
            A list of category names
        """
        # List all directories in the base directory
        categories = []
        try:
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path):
                    categories.append(item)
        except Exception as e:
            logger.error(f"Error listing categories: {str(e)}")
        
        return categories
    
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

