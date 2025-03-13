from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from app.config import settings

# Create SQLAlchemy engine
# For SQLite, we need to add check_same_thread=False
# For other databases, we don't need this parameter
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL, 
    connect_args=connect_args
)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()

# Create a function to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

