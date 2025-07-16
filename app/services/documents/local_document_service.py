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
    
    def _get_document_path(self, category: str, doc_id: str, ext: str = "json") -> str:
        """Get the path to a document file with the given extension."""
        return os.path.join(self._get_category_path(category), f"{doc_id}.{ext}")
    
    def get_document(self, category: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by category and ID.
        Tries both .json and .md extensions.
        """
        # Try JSON first
        doc_path_json = self._get_document_path(category, doc_id, "json")
        doc_path_md = self._get_document_path(category, doc_id, "md")
        
        if os.path.exists(doc_path_json):
            try:
                with open(doc_path_json, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                    return document
            except Exception as e:
                logger.error(f"Error reading JSON document {doc_id}: {str(e)}")
                return None
        elif os.path.exists(doc_path_md):
            try:
                with open(doc_path_md, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Return as a dict for consistency
                return {
                    "id": doc_id,
                    "category": category,
                    "content": content,
                    "content_type": "markdown"
                }
            except Exception as e:
                logger.error(f"Error reading Markdown document {doc_id}: {str(e)}")
                return None
        else:
            logger.warning(f"No document found with ID {doc_id} in category {category}")
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
                elif filename.endswith('.md'):
                    doc_id = filename.replace('.md', '')

                    # Get the document
                    document = self.get_document(cat, doc_id)
                    if document:
                        # Add category and ID to the document
                        document['category'] = cat
                        document['id'] = doc_id
                        document['title'] = doc_id
                        document['created_at'] = datetime.utcnow().isoformat()
                        document['updated_at'] = datetime.utcnow().isoformat()
                        documents.append(document)
                else:
                    pass
        
        return documents
    
    def add_document(self, category: str, doc_data: Dict[str, Any]) -> str:
        """
        Add a new document to the specified category.
        Supports both JSON and Markdown documents.
        """
        doc_id = doc_data.get('id', str(uuid.uuid4()))
        doc_data['id'] = doc_id
        doc_data['category'] = category
        doc_data['created_at'] = datetime.utcnow().isoformat()
        doc_data['updated_at'] = datetime.utcnow().isoformat()

        content_type = doc_data.get("content_type", "json").lower()
        if content_type in ["markdown", "md"]:
            doc_path = self._get_document_path(category, doc_id, "md")
            try:
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(doc_data.get("content", ""))
                return doc_id
            except Exception as e:
                logger.error(f"Error adding Markdown document: {str(e)}")
                raise ValueError(f"Failed to add Markdown document: {str(e)}")
        else:
            doc_path = self._get_document_path(category, doc_id, "json")
            try:
                with open(doc_path, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, ensure_ascii=False, indent=2)
                return doc_id
            except Exception as e:
                logger.error(f"Error adding JSON document: {str(e)}")
                raise ValueError(f"Failed to add JSON document: {str(e)}")

    def update_document(self, category: str, doc_id: str, doc_data: Dict[str, Any]) -> bool:
        """
        Update an existing document.
        Supports both JSON and Markdown documents.
        """
        content_type = doc_data.get("content_type", "json").lower()
        if content_type in ["markdown", "md"]:
            doc_path = self._get_document_path(category, doc_id, "md")
            try:
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(doc_data.get("content", ""))
                return True
            except Exception as e:
                logger.error(f"Error updating Markdown document {doc_id}: {str(e)}")
                return False
        else:
            doc_path = self._get_document_path(category, doc_id, "json")
            try:
                existing_doc = self.get_document(category, doc_id) or {}
                doc_data['id'] = doc_id
                doc_data['category'] = category
                doc_data['created_at'] = existing_doc.get('created_at', datetime.utcnow().isoformat())
                doc_data['updated_at'] = datetime.utcnow().isoformat()
                with open(doc_path, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                logger.error(f"Error updating JSON document {doc_id}: {str(e)}")
                return False

    def delete_document(self, category: str, doc_id: str) -> bool:
        """
        Delete a document.
        Deletes both .json and .md files if they exist.
        """
        deleted = False
        for ext in ["json", "md"]:
            doc_path = self._get_document_path(category, doc_id, ext)
            if os.path.exists(doc_path):
                try:
                    os.remove(doc_path)
                    deleted = True
                except Exception as e:
                    logger.error(f"Error deleting document {doc_id} ({ext}): {str(e)}")
        return deleted
    
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

