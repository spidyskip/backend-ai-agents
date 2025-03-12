import logging
from typing import Dict, List, Optional, Any, Union, cast, Iterable
from sqlalchemy.orm import Session
import uuid
import re
from collections import Counter
from pydantic import SecretStr

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.models import AgentConfig
from app.tools import AVAILABLE_TOOLS
from app.config import settings

logger = logging.getLogger(__name__)

class AgentManager:
    # Store agent graphs and their checkpointers
    agents = {}
    checkpointers = {}
    # Store agent metadata for selection
    agent_metadata = {}
    
    @staticmethod
    def create_agent(
        agent_id: str, 
        name: str,
        prompt: str, 
        model_name: str,
        tool_names: List[str],
        db: Session,
        categories: List[str] = [],
        keywords: List[str] = []
    ):
        # Check if agent already exists in memory
        if agent_id in AgentManager.agents:
            raise ValueError(f"Agent with ID '{agent_id}' already exists in memory.")
        
        # Check if agent already exists in database
        existing_agent = db.query(AgentConfig).filter(AgentConfig.id == agent_id).first()
        if existing_agent:
            # If agent exists in database but not in memory, load it into memory
            logger.info(f"Agent with ID '{agent_id}' already exists in database. Loading into memory.")
            
            # Extract values from SQLAlchemy columns and convert to appropriate types
            agent_id_str = str(existing_agent.id)
            name_str = str(existing_agent.name)
            prompt_str = str(existing_agent.prompt)
            model_name_str = str(existing_agent.model_name)
            
            # Handle JSON columns that might be None
            tools_list = []
            if existing_agent.tools is not None:
                # Convert SQLAlchemy Column to list
                if hasattr(existing_agent.tools, '__iter__'):
                    tools_list = list(existing_agent.tools)
                else:
                    # If it's not iterable, convert to string and parse
                    tools_str = str(existing_agent.tools)
                    if tools_str.startswith('[') and tools_str.endswith(']'):
                        # Parse JSON-like string
                        tools_list = [item.strip(' "\'') for item in tools_str[1:-1].split(',') if item.strip()]
            
            categories_list = []
            if existing_agent.categories is not None:
                # Convert SQLAlchemy Column to list
                if hasattr(existing_agent.categories, '__iter__'):
                    categories_list = list(existing_agent.categories)
                else:
                    # If it's not iterable, convert to string and parse
                    categories_str = str(existing_agent.categories)
                    if categories_str.startswith('[') and categories_str.endswith(']'):
                        # Parse JSON-like string
                        categories_list = [item.strip(' "\'') for item in categories_str[1:-1].split(',') if item.strip()]
            
            keywords_list = []
            if existing_agent.keywords is not None:
                # Convert SQLAlchemy Column to list
                if hasattr(existing_agent.keywords, '__iter__'):
                    keywords_list = list(existing_agent.keywords)
                else:
                    # If it's not iterable, convert to string and parse
                    keywords_str = str(existing_agent.keywords)
                    if keywords_str.startswith('[') and keywords_str.endswith(']'):
                        # Parse JSON-like string
                        keywords_list = [item.strip(' "\'') for item in keywords_str[1:-1].split(',') if item.strip()]
            
            # Select the appropriate model
            if "claude" in model_name_str.lower():
                model = ChatAnthropic(
                    model=model_name_str,
                    temperature=0,
                    api_key=settings.ANTHROPIC_API_KEY
                )
            elif "gemini" in model_name_str.lower():
                model = ChatGoogleGenerativeAI(
                    model=model_name_str,
                    temperature=0,
                    api_key=settings.GOOGLE_API_KEY
                )
            else:
                model = ChatOpenAI(
                    model=model_name_str,
                    temperature=0,
                    api_key=settings.OPENAI_API_KEY
                )
            
            # Get the requested tools
            tools = [AVAILABLE_TOOLS[tool_name] for tool_name in tools_list if tool_name in AVAILABLE_TOOLS]
            
            # Initialize memory to persist state between graph runs
            checkpointer = MemorySaver()
            
            # Create the agent using LangGraph's prebuilt function
            agent_graph = create_react_agent(
                model=model, 
                tools=tools, 
                checkpointer=checkpointer,
                prompt=prompt_str
            )
            
            # Store the agent and its checkpointer
            AgentManager.agents[agent_id_str] = agent_graph
            AgentManager.checkpointers[agent_id_str] = checkpointer
            
            # Store agent metadata for selection
            AgentManager.agent_metadata[agent_id_str] = {
                "name": name_str,
                "prompt": prompt_str,
                "categories": categories_list,
                "keywords": keywords_list
            }
            
            return {
                "agent_id": agent_id_str,
                "name": name_str,
                "prompt": prompt_str,
                "model_name": model_name_str,
                "tools": tools_list,
                "categories": categories_list,
                "keywords": keywords_list
            }
        
        # If agent doesn't exist, create a new one
        # Select the appropriate model
        if "claude" in model_name.lower():
            model = ChatAnthropic(
                model=model_name,
                temperature=0,
                timeout=None,
                stop=None,  
                api_key=settings.ANTHROPIC_API_KEY
            )
        elif "gemini" in model_name.lower():
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                api_key=settings.GOOGLE_API_KEY
            )
        else:
            model = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=settings.OPENAI_API_KEY
            )
        
        # Get the requested tools
        tools = [AVAILABLE_TOOLS[tool_name] for tool_name in tool_names if tool_name in AVAILABLE_TOOLS]
        
        # Initialize memory to persist state between graph runs
        checkpointer = MemorySaver()
        
        # Create the agent using LangGraph's prebuilt function
        agent_graph = create_react_agent(
            model=model, 
            tools=tools, 
            checkpointer=checkpointer,
            prompt=prompt
        )
        
        # Store the agent and its checkpointer
        AgentManager.agents[agent_id] = agent_graph
        AgentManager.checkpointers[agent_id] = checkpointer
        
        # Store agent metadata for selection
        AgentManager.agent_metadata[agent_id] = {
            "name": name,
            "prompt": prompt,
            "categories": categories,
            "keywords": keywords
        }
        
        # Persist to database
        db_agent = AgentConfig(
            id=agent_id, 
            name=name,
            prompt=prompt, 
            model_name=model_name,
            tools=tool_names,
            categories=categories,
            keywords=keywords
        )
        db.add(db_agent)
        db.commit()
        
        return {
            "agent_id": agent_id,
            "name": name,
            "prompt": prompt,
            "model_name": model_name,
            "tools": tool_names,
            "categories": categories,
            "keywords": keywords
        }
    
    @staticmethod
    def get_agent(agent_id: str, db: Session):
        # Check if agent is already loaded
        if agent_id not in AgentManager.agents:
            # Try to load from database
            db_agent = db.query(AgentConfig).filter(AgentConfig.id == agent_id).first()
            if not db_agent:
                raise ValueError(f"No agent found with ID '{agent_id}'.")
            
            # Extract values from SQLAlchemy columns and convert to appropriate types
            agent_id_str = str(db_agent.id)
            name_str = str(db_agent.name)
            prompt_str = str(db_agent.prompt)
            model_name_str = str(db_agent.model_name)
            
            # Handle JSON columns that might be None
            tools_list = []
            if db_agent.tools is not None:
                # Convert SQLAlchemy Column to list
                if hasattr(db_agent.tools, '__iter__'):
                    tools_list = list(db_agent.tools)
                else:
                    # If it's not iterable, convert to string and parse
                    tools_str = str(db_agent.tools)
                    if tools_str.startswith('[') and tools_str.endswith(']'):
                        # Parse JSON-like string
                        tools_list = [item.strip(' "\'') for item in tools_str[1:-1].split(',') if item.strip()]
            
            categories_list = []
            if db_agent.categories is not None:
                # Convert SQLAlchemy Column to list
                if hasattr(db_agent.categories, '__iter__'):
                    categories_list = list(db_agent.categories)
                else:
                    # If it's not iterable, convert to string and parse
                    categories_str = str(db_agent.categories)
                    if categories_str.startswith('[') and categories_str.endswith(']'):
                        # Parse JSON-like string
                        categories_list = [item.strip(' "\'') for item in categories_str[1:-1].split(',') if item.strip()]
            
            keywords_list = []
            if db_agent.keywords is not None:
                # Convert SQLAlchemy Column to list
                if hasattr(db_agent.keywords, '__iter__'):
                    keywords_list = list(db_agent.keywords)
                else:
                    # If it's not iterable, convert to string and parse
                    keywords_str = str(db_agent.keywords)
                    if keywords_str.startswith('[') and keywords_str.endswith(']'):
                        # Parse JSON-like string
                        keywords_list = [item.strip(' "\'') for item in keywords_str[1:-1].split(',') if item.strip()]
            
            # Recreate the agent
            AgentManager.create_agent(
                agent_id_str,
                name_str,
                prompt_str,
                model_name_str,
                tools_list,
                db,
                categories_list,
                keywords_list
            )
        
        return {
            "agent": AgentManager.agents[agent_id],
            "checkpointer": AgentManager.checkpointers[agent_id],
            "metadata": AgentManager.agent_metadata[agent_id]
        }
    
    @staticmethod
    def list_agents(db: Session):
        # Get all agents from database
        db_agents = db.query(AgentConfig).all()
        return [
            {
                "agent_id": agent.id,
                "name": agent.name,
                "prompt": agent.prompt,
                "model_name": agent.model_name,
                "tools": list(agent.tools) if agent.tools is not None and hasattr(agent.tools, '__iter__') else [],
                "categories": list(agent.categories) if agent.categories is not None and hasattr(agent.categories, '__iter__') else [],
                "keywords": list(agent.keywords) if agent.keywords is not None and hasattr(agent.keywords, '__iter__') else []
            }
            for agent in db_agents
        ]
    
    @staticmethod
    def load_agents_from_db(db: Session):
        db_agents = db.query(AgentConfig).all()
        loaded_count = 0
        
        for db_agent in db_agents:
            if str(db_agent.id) not in AgentManager.agents:
                try:
                    # Extract values from SQLAlchemy columns and convert to appropriate types
                    agent_id_str = str(db_agent.id)
                    name_str = str(db_agent.name)
                    prompt_str = str(db_agent.prompt)
                    model_name_str = str(db_agent.model_name)
                    
                    # Handle JSON columns that might be None
                    tools_list = []
                    if db_agent.tools is not None:
                        # Convert SQLAlchemy Column to list
                        if hasattr(db_agent.tools, '__iter__'):
                            tools_list = list(db_agent.tools)
                        else:
                            # If it's not iterable, convert to string and parse
                            tools_str = str(db_agent.tools)
                            if tools_str.startswith('[') and tools_str.endswith(']'):
                                # Parse JSON-like string
                                tools_list = [item.strip(' "\'') for item in tools_str[1:-1].split(',') if item.strip()]
                    
                    categories_list = []
                    if db_agent.categories is not None:
                        # Convert SQLAlchemy Column to list
                        if hasattr(db_agent.categories, '__iter__'):
                            categories_list = list(db_agent.categories)
                        else:
                            # If it's not iterable, convert to string and parse
                            categories_str = str(db_agent.categories)
                            if categories_str.startswith('[') and categories_str.endswith(']'):
                                # Parse JSON-like string
                                categories_list = [item.strip(' "\'') for item in categories_str[1:-1].split(',') if item.strip()]
                    
                    keywords_list = []
                    if db_agent.keywords is not None:
                        # Convert SQLAlchemy Column to list
                        if hasattr(db_agent.keywords, '__iter__'):
                            keywords_list = list(db_agent.keywords)
                        else:
                            # If it's not iterable, convert to string and parse
                            keywords_str = str(db_agent.keywords)
                            if keywords_str.startswith('[') and keywords_str.endswith(']'):
                                # Parse JSON-like string
                                keywords_list = [item.strip(' "\'') for item in keywords_str[1:-1].split(',') if item.strip()]
                    
                    AgentManager.create_agent(
                        agent_id_str,
                        name_str,
                        prompt_str,
                        model_name_str,
                        tools_list,
                        db,
                        categories_list,
                        keywords_list
                    )
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Error loading agent {db_agent.id}: {str(e)}")
        
        logger.info(f"Loaded {loaded_count} agents from database")
    
    @staticmethod
    def select_agent_for_query(query: str, db: Session):
        """
        Select the most appropriate agent for a given query.
        This uses a simple keyword matching algorithm, but could be replaced
        with a more sophisticated approach like embeddings similarity.
        """
        if not AgentManager.agent_metadata:
            # Load agents if not already loaded
            AgentManager.load_agents_from_db(db)
            
        if not AgentManager.agent_metadata:
            raise ValueError("No agents available to handle the query.")
        
        # If only one agent exists, use it
        if len(AgentManager.agent_metadata) == 1:
            agent_id = next(iter(AgentManager.agent_metadata))
            return {
                "agent_id": agent_id,
                "confidence": 1.0,
                "name": AgentManager.agent_metadata[agent_id]["name"]
            }
        
        # Simple keyword matching
        query_lower = query.lower()
        scores = {}
        
        for agent_id, metadata in AgentManager.agent_metadata.items():
            score = 0
            
            # Check keywords
            for keyword in metadata.get("keywords", []):
                if keyword.lower() in query_lower:
                    score += 2  # Keywords are more important
            
            # Check categories
            for category in metadata.get("categories", []):
                if category.lower() in query_lower:
                    score += 1
            
            # Store the score
            scores[agent_id] = score
        
        # Get the agent with the highest score
        if not scores or all(score == 0 for score in scores.values()):
            # If no matches, use a default agent (first one)
            agent_id = next(iter(AgentManager.agent_metadata))
            confidence = 0.5  # Low confidence
        else:
            # Find the key with the maximum value using a different approach
            max_score = -1
            max_agent_id = None
            for agent_id, score in scores.items():
                if score > max_score:
                    max_score = score
                    max_agent_id = agent_id
            
            agent_id = max_agent_id
            confidence = min(1.0, scores[agent_id] / 5.0)  # Normalize confidence
        
        return {
            "agent_id": agent_id,
            "confidence": confidence,
            "name": AgentManager.agent_metadata[agent_id]["name"]
        }
    
    @staticmethod
    async def process_chat(
        query: str,
        agent_id: Optional[str],
        thread_id: str,
        db: Session
    ):
        """
        Process a chat query using the appropriate agent.
        If agent_id is provided, that specific agent will be used.
        If not, the system will select the most appropriate agent based on the query.
        """
        # If no agent_id provided, select the most appropriate agent
        if not agent_id:
            selected_agent = AgentManager.select_agent_for_query(query, db)
            agent_id = selected_agent["agent_id"]
            agent_name = selected_agent["name"]
            confidence = selected_agent["confidence"]
        else:
            # Check if the agent exists
            if agent_id not in AgentManager.agents:
                # Try to load from database
                db_agent = db.query(AgentConfig).filter(AgentConfig.id == agent_id).first()
                if not db_agent:
                    raise ValueError(f"No agent found with ID '{agent_id}'.")
                
                # Load the agent
                AgentManager.get_agent(agent_id, db)
            
            agent_name = AgentManager.agent_metadata[agent_id]["name"]
            confidence = 1.0  # User explicitly selected this agent
        
        # Get the agent
        agent_info = AgentManager.get_agent(agent_id, db)
        agent = agent_info["agent"]
        
        # Create input for the agent
        agent_input = {
            "messages": [{"role": "user", "content": query}]
        }
        
        # Invoke the agent with the thread_id for state persistence
        final_state = agent.invoke(
            agent_input,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract the response
        response = final_state["messages"][-1].content
        
        return {
            "response": response,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "thread_id": thread_id,
            "confidence": confidence
        }

