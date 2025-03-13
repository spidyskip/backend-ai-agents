import sys
import os
import logging

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_db_service
from app.services.agent_manager import AgentManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database with sample agents"""
    try:
        # Get database service
        db_service = get_db_service()
        
        agents = {}
        
        # Check if agents already exist
        existing_agents = db_service.list_agents()
        if existing_agents:
            logger.info(f"Found {len(existing_agents)} existing agents in the database.")
            for agent in existing_agents:
                logger.info(f"Agent {agent['id']} already exists.")
            
            # Load existing agents into memory
            AgentManager.load_agents_from_db()
            return {agent["id"]: {"name": agent["name"]} for agent in existing_agents}
        
        # Create a weather agent
        try:
            weather_agent = AgentManager.create_agent(
                agent_id="weather-agent",
                name="Weather Assistant",
                prompt="You are a helpful assistant that specializes in providing weather information. Always be concise and accurate.",
                model_name="gemini-2.0-flash",
                tool_names=["search"],
                categories=["weather", "climate", "forecast"],
                keywords=["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", "humidity", "wind"]
            )
            logger.info(f"Created weather agent: {weather_agent['name']}")
            agents["weather_agent"] = weather_agent
        except ValueError as e:
            logger.warning(f"Could not create weather agent: {str(e)}")
        
        # Create a math agent
        try:
            math_agent = AgentManager.create_agent(
                agent_id="math-agent",
                name="Math Solver",
                prompt="You are a math expert that helps solve mathematical problems. Show your work step by step.",
                model_name="gemini-2.0-flash",
                tool_names=["calculator"],
                categories=["math", "calculation", "algebra", "geometry"],
                keywords=["calculate", "solve", "equation", "math", "formula", "computation", "problem"]
            )
            logger.info(f"Created math agent: {math_agent['name']}")
            agents["math_agent"] = math_agent
        except ValueError as e:
            logger.warning(f"Could not create math agent: {str(e)}")
        
        # Create a general knowledge agent
        try:
            general_agent = AgentManager.create_agent(
                agent_id="general-agent",
                name="General Knowledge Assistant",
                prompt="You are a helpful assistant with broad knowledge. Provide accurate and informative responses.",
                model_name="gemini-2.0-flash",
                tool_names=["search"],
                categories=["general", "knowledge", "information", "news"],
                keywords=["what", "who", "when", "where", "why", "how", "explain", "describe", "tell"]
            )
            logger.info(f"Created general agent: {general_agent['name']}")
            agents["general_agent"] = general_agent
        except ValueError as e:
            logger.warning(f"Could not create general agent: {str(e)}")
        
        # Create a Gemini agent
        try:
            gemini_agent = AgentManager.create_agent(
                agent_id="gemini-agent",
                name="Gemini Assistant",
                prompt="You are a helpful assistant powered by Google's Gemini model. Provide accurate and informative responses.",
                model_name="gemini-2.0-flash",
                tool_names=["search", "calculator", "database_lookup"],
                categories=["general", "knowledge", "information"],
                keywords=["gemini", "google", "ai", "assistant"]
            )
            logger.info(f"Created Gemini agent: {gemini_agent['name']}")
            agents["gemini_agent"] = gemini_agent
        except ValueError as e:
            logger.warning(f"Could not create Gemini agent: {str(e)}")
        
        return agents
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return None

if __name__ == "__main__":
    result = init_db()
    if result:
        logger.info("Database initialized successfully with sample agents")
    else:
        logger.error("Failed to initialize database")

