from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

class DocumentInterface(ABC):
    """Abstract interface for document operations."""
    
    @abstractmethod
    def get_document(self, category: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by category and ID."""
        pass
    
    @abstractmethod
    def list_documents(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all documents, optionally filtered by category."""
        pass
    
    @abstractmethod
    def add_document(self, category: str, doc_data: Dict[str, Any]) -> str:
        """Add a new document to the specified category."""
        pass
    
    @abstractmethod
    def update_document(self, category: str, doc_id: str, doc_data: Dict[str, Any]) -> bool:
        """Update an existing document."""
        pass
    
    @abstractmethod
    def delete_document(self, category: str, doc_id: str) -> bool:
        """Delete a document."""
        pass
    
    @abstractmethod
    def list_categories(self) -> List[str]:
        """List all available document categories."""
        pass
    
    @abstractmethod
    def get_documents_by_ids(self, category: str, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents by their IDs within a category."""
        pass

