import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Dynamic Agent Backend"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "A robust backend system for creating and managing AI agents using LangGraph and FastAPI"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./agents.db"
    
    # API Keys for various services
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    
    # Default model settings
    DEFAULT_LLM_MODEL: str = "gemini-2.0-flash"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

# Create global settings object
settings = Settings()

