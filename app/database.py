from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from typing import Optional, Union

from app.config import settings, DatabaseType
from app.services.database.interface import DatabaseInterface

# Create SQLAlchemy engine and session for SQLite
if settings.DATABASE_TYPE == DatabaseType.SQLITE:
    # For SQLite, we need to add check_same_thread=False
    connect_args = {"check_same_thread": False}
    engine = create_engine(
        settings.database_url, 
        connect_args=connect_args
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    # For MongoDB, we don't use SQLAlchemy
    engine = None
    SessionLocal = None

# Create declarative base for SQLAlchemy models
Base = declarative_base()

# Database service instance
db_service: Optional[DatabaseInterface] = None

def get_db_session():
    """Get a database session for SQLite."""
    if settings.DATABASE_TYPE == DatabaseType.SQLITE:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        yield None

def get_db_service() -> DatabaseInterface:
    """Get the appropriate database service based on configuration."""
    global db_service
    
    if db_service is not None:
        return db_service
    
    if settings.DATABASE_TYPE == DatabaseType.SQLITE:
        from app.services.database.sqlite_service import SQLiteService
        db = next(get_db_session())
        db_service = SQLiteService(db)
    elif settings.DATABASE_TYPE == DatabaseType.MONGODB:
        from app.services.database.mongodb_service import MongoDBService
        db_service = MongoDBService()
    else:
        raise ValueError(f"Unsupported database type: {settings.DATABASE_TYPE}")
    
    return db_service

