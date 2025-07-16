import logging
from typing import Dict, List, Optional, Any, Union

from app.config import settings
from app.services.documents.interface import DocumentInterface

logger = logging.getLogger(__name__)

# Document service instance
document_service: Optional[DocumentInterface] = None

def get_document_service() -> DocumentInterface:
    """Get the appropriate document service based on configuration."""
    global document_service
    
    if document_service is not None:
        return document_service
    
    if getattr(settings, "DOCUMENT_STORAGE_TYPE", None) == "S3":
        from app.services.documents.s3_document_service import S3DocumentService
        document_service = S3DocumentService()
    else:
        from app.services.documents.local_document_service import LocalDocumentService
        document_service = LocalDocumentService()
    
    return document_service

