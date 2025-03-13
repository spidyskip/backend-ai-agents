import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from enum import Enum

# Load environment variables from .env file
load_dotenv()

class DatabaseType(str, Enum):
    SQLITE = "sqlite"
    MONGODB = "mongodb"

class Settings(BaseSettings):
    APP_NAME: str = "Dynamic Agent Backend"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "A robust backend system for creating and managing AI agents using LangGraph and FastAPI"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Database settings
    DATABASE_TYPE: DatabaseType = os.getenv("DATABASE_TYPE", DatabaseType.SQLITE)
    
    # SQLite settings (used when DATABASE_TYPE is sqlite)
    SQLITE_URL: str = os.getenv("SQLITE_URL", "sqlite:///./agents.db")
    
    # MongoDB settings (used when DATABASE_TYPE is mongodb)
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "agents_db")
    
    # S3 settings for storing agent prompts and other data
    USE_S3_STORAGE: bool = os.getenv("USE_S3_STORAGE", "false").lower() == "true"
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    
    # API Keys for various services
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""  # Added for Gemini
    
    # Default model settings
    DEFAULT_LLM_MODEL: str = "gpt-4o"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
    
    @property
    def database_url(self) -> str:
        """Get the appropriate database URL based on the database type."""
        if self.DATABASE_TYPE == DatabaseType.SQLITE:
            return self.SQLITE_URL
        return ""  # For MongoDB, we use the URI directly

# Create global settings object
settings = Settings()

