import logging
from typing import Dict, List, Optional, Any, Union, cast, Iterable
import uuid
import re
from collections import Counter
from pydantic import SecretStr
from datetime import datetime
import asyncio

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings, DatabaseType
from app.tools import AVAILABLE_TOOLS
from app.database import get_db_service
from app.services.database.interface import DatabaseInterface
from app.services.documents.documents_manager import get_document_service
from app.services.supervisor_agent import SupervisorAgent

logger = logging.getLogger(__name__)

class AgentManager:
    # Store agent graphs and their checkpointers
    agents = {}
    checkpointers = {}
    # Store agent metadata for selection
    agent_metadata = {}
    # Store additional information for agents
    agent_additional_query = {}

    supervisor_agent: Optional[SupervisorAgent] = SupervisorAgent() if settings.ENABLE_SUPERVISOR_AGENT else None
    
    @staticmethod
    def create_agent(
        agent_id: str, 
        name: str,
        prompt: str, 
        model_name: str,
        tool_names: List[str],
        categories: List[str] = [],
        keywords: List[str] = [],
        additional_query: Dict[str, Any] = {},
        document_refs: Dict[str, List[str]] = {}
    ):
        # Check if agent already exists in memory
        if agent_id in AgentManager.agents:
            raise ValueError(f"Agent with ID '{agent_id}' already exists in memory.")
        
        # Get database service
        db_service = get_db_service()
        
        # Check if agent already exists in database
        existing_agent = db_service.get_agent(agent_id)
        
        if existing_agent:
            # If agent exists in database but not in memory, load it into memory
            logger.info(f"Agent with ID '{agent_id}' already exists in database. Loading into memory.")
            
            # Extract values from database
            agent_id_str = existing_agent["id"]
            name_str = existing_agent["name"]
            prompt_str = existing_agent["prompt"]
            model_name_str = existing_agent["model_name"]
            tools_list = existing_agent.get("tools", [])
            categories_list = existing_agent.get("categories", [])
            keywords_list = existing_agent.get("keywords", [])
            additional_query_dict = existing_agent.get("additional_query", {})
            document_refs_dict = existing_agent.get("document_refs", {})
            
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
            tools = []
            for tool_name in tools_list:
                if tool_name in AVAILABLE_TOOLS:
                    tools.append(AVAILABLE_TOOLS[tool_name])
            
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
                "keywords": keywords_list,
                "document_refs": document_refs_dict
            }
            
            # Store additional information
            AgentManager.agent_additional_query[agent_id_str] = additional_query_dict
            
            return {
                "agent_id": agent_id_str,
                "name": name_str,
                "prompt": prompt_str,
                "model_name": model_name_str,
                "tools": tools_list,
                "categories": categories_list,
                "keywords": keywords_list,
                "additional_query": additional_query_dict,
                "document_refs": document_refs_dict
            }
        
        # If agent doesn't exist, create a new one
        # Select the appropriate model
        if "claude" in model_name.lower():
            model = ChatAnthropic(
                model=model_name,
                temperature=0,
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
        tools = []
        for tool_name in tool_names:
            if tool_name in AVAILABLE_TOOLS:
                tools.append(AVAILABLE_TOOLS[tool_name])
        
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
            "keywords": keywords,
            "document_refs": document_refs
        }
        
        # Store additional information
        AgentManager.agent_additional_query[agent_id] = additional_query
        
        # Persist to database
        db_service.create_agent({
            "id": agent_id,
            "name": name,
            "prompt": prompt,
            "model_name": model_name,
            "tools": tool_names,
            "categories": categories,
            "keywords": keywords,
            "additional_query": additional_query,
            "document_refs": document_refs
        })
        
        return {
            "agent_id": agent_id,
            "name": name,
            "prompt": prompt,
            "model_name": model_name,
            "tools": tool_names,
            "categories": categories,
            "keywords": keywords,
            "additional_query": additional_query,
            "document_refs": document_refs
        }
    
    @staticmethod
    def update_agent(agent_id: str, update_data: Dict[str, Any]):
        """Update an existing agent with new data."""
        # Check if agent exists
        if agent_id not in AgentManager.agents:
            # Try to load from database or S3
            try:
                AgentManager.get_agent(agent_id)
            except ValueError:
                raise ValueError(f"No agent found with ID '{agent_id}'.")
        
        # Get database service
        db_service = get_db_service()
        
        # Get agent from database
        db_agent = db_service.get_agent(agent_id)
        if not db_agent:
            raise ValueError(f"No agent found with ID '{agent_id}' in database.")
        
        # Update agent in database
        updated_agent = {**db_agent}
        
        # Update fields if provided
        if "name" in update_data:
            updated_agent["name"] = update_data["name"]
        if "prompt" in update_data:
            updated_agent["prompt"] = update_data["prompt"]
        if "model_name" in update_data:
            updated_agent["model_name"] = update_data["model_name"]
        if "tools" in update_data:
            updated_agent["tools"] = update_data["tools"]
        if "categories" in update_data:
            updated_agent["categories"] = update_data["categories"]
        if "keywords" in update_data:
            updated_agent["keywords"] = update_data["keywords"]
        if "additional_query" in update_data:
            updated_agent["additional_query"] = update_data["additional_query"]
        if "document_refs" in update_data:
            updated_agent["document_refs"] = update_data["document_refs"]
        
        # Update in database
        db_service.update_agent(agent_id, updated_agent)
        
        # Recreate the agent in memory
        # Remove the old agent
        if agent_id in AgentManager.agents:
            del AgentManager.agents[agent_id]
        if agent_id in AgentManager.checkpointers:
            del AgentManager.checkpointers[agent_id]
        if agent_id in AgentManager.agent_metadata:
            del AgentManager.agent_metadata[agent_id]
        if agent_id in AgentManager.agent_additional_query:
            del AgentManager.agent_additional_query[agent_id]
        
        # Create the updated agent
        AgentManager.create_agent(
            agent_id,
            updated_agent["name"],
            updated_agent["prompt"],
            updated_agent["model_name"],
            updated_agent["tools"],
            updated_agent.get("categories", []),
            updated_agent.get("keywords", []),
            updated_agent.get("additional_query", {}),
            updated_agent.get("document_refs", {})
        )
        
        return {
            "agent_id": agent_id,
            "name": updated_agent["name"],
            "prompt": updated_agent["prompt"],
            "model_name": updated_agent["model_name"],
            "tools": updated_agent["tools"],
            "categories": updated_agent.get("categories", []),
            "keywords": updated_agent.get("keywords", []),
            "additional_query": updated_agent.get("additional_query", {}),
            "document_refs": updated_agent.get("document_refs", {})
        }
    
    @staticmethod
    def get_agent(agent_id: str):
        # Check if agent is already loaded
        if agent_id not in AgentManager.agents:
            # Get database service
            db_service = get_db_service()
            
            # Try to load from database or S3
            db_agent = db_service.get_agent(agent_id)
            
            if not db_agent:             
                raise ValueError(f"No agent found with ID '{agent_id}'.")
            
            # Extract values from database
            agent_id_str = db_agent["id"]
            name_str = db_agent["name"]
            prompt_str = db_agent["prompt"]
            model_name_str = db_agent["model_name"]
            tools_list = db_agent.get("tools", [])
            categories_list = db_agent.get("categories", [])
            keywords_list = db_agent.get("keywords", [])
            additional_query_dict = db_agent.get("additional_query", {})
            document_refs_dict = db_agent.get("document_refs", {})
            
            # Recreate the agent
            AgentManager.create_agent(
                agent_id_str,
                name_str,
                prompt_str,
                model_name_str,
                tools_list,
                categories_list,
                keywords_list,
                additional_query_dict,
                document_refs_dict
            )
        
        return {
            "agent": AgentManager.agents[agent_id],
            "checkpointer": AgentManager.checkpointers[agent_id],
            "metadata": AgentManager.agent_metadata[agent_id],
            "additional_query": AgentManager.agent_additional_query.get(agent_id, {})
        }
    
    @staticmethod
    def update_agent_additional_query(agent_id: str, additional_query: Dict[str, Any]):
        """Update the additional information for an agent."""
        # Check if agent exists
        if agent_id not in AgentManager.agents:
            # Try to load from database or S3
            AgentManager.get_agent(agent_id)
        
        # Update additional info
        AgentManager.agent_additional_query[agent_id] = additional_query
        
        # Get database service
        db_service = get_db_service()
        
        # Get agent from database
        db_agent = db_service.get_agent(agent_id)
        if db_agent:
            # Update agent in database
            db_agent["additional_query"] = additional_query
            db_service.update_agent(agent_id, db_agent)
        
        return additional_query
    
    @staticmethod
    def update_agent_document_refs(agent_id: str, document_refs: Dict[str, List[str]]):
        """Update the document references for an agent."""
        # Check if agent exists
        if agent_id not in AgentManager.agents:
            # Try to load from database or S3
            AgentManager.get_agent(agent_id)
        
        # Update document refs in metadata
        if agent_id in AgentManager.agent_metadata:
            AgentManager.agent_metadata[agent_id]["document_refs"] = document_refs
        
        # Get database service
        db_service = get_db_service()
        
        # Get agent from database
        db_agent = db_service.get_agent(agent_id)
        if db_agent:
            # Update agent in database
            db_agent["document_refs"] = document_refs
            db_service.update_agent(agent_id, db_agent)
        
        return document_refs
    
    @staticmethod
    def list_agents():
        # Get database service
        db_service = get_db_service()
        
        # Get all agents from database
        db_agents = db_service.list_agents()
        
        return [
            {
                "agent_id": agent["id"],
                "name": agent["name"],
                "prompt": agent["prompt"],
                "model_name": agent["model_name"],
                "tools": agent.get("tools", []),
                "categories": agent.get("categories", []),
                "keywords": agent.get("keywords", []),
                "additional_query": agent.get("additional_query", {}),
                "document_refs": agent.get("document_refs", {})
            }
            for agent in db_agents
        ]
    
    @staticmethod
    def load_agents_from_db():
        # Get database service
        db_service = get_db_service()
        
        # Get all agents from database
        db_agents = db_service.list_agents()
        loaded_count = 0
        
        for db_agent in db_agents:
            if db_agent["id"] not in AgentManager.agents:
                try:
                    AgentManager.create_agent(
                        db_agent["id"],
                        db_agent["name"],
                        db_agent["prompt"],
                        db_agent["model_name"],
                        db_agent.get("tools", []),
                        db_agent.get("categories", []),
                        db_agent.get("keywords", []),
                        db_agent.get("additional_query", {}),
                        db_agent.get("document_refs", {})
                    )
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Error loading agent {db_agent['id']}: {str(e)}")
        
        logger.info(f"Loaded {loaded_count} agents from storage")
    
    @staticmethod
    def select_agent_for_query(query: str):
        """
        Select the most appropriate agent for a given query.
        This uses a simple keyword matching algorithm, but could be replaced
        with a more sophisticated approach like embeddings similarity.
        """
        if not AgentManager.agent_metadata:
            # Load agents if not already loaded
            AgentManager.load_agents_from_db()
            
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
        if AgentManager.supervisor_agent != None:
            agents = {
                key: {k: v for k, v in value.items() if k != "prompt"}
                for key, value in AgentManager.agent_metadata.items()
            }
            try:
                result = AgentManager.supervisor_agent.select_agent(query=query_lower, thread_id="supervisor", agents=agents)
                
                if result.get("response") in AgentManager.agent_metadata:  # Check if 'response' exists in result
                    logger.info(f"Supervisor agent selected agent {result.get('response')}")
                    agent_id = result.get("response")
                    return {
                            "agent_id": agent_id,
                            "confidence": 0,
                            "name": AgentManager.agent_metadata[agent_id]["name"]
                        }
            except Exception as e:
                logger.error(f"Error invoking supervisor agent: {e}")
                pass
        
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
            # Find the key with the maximum value
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
        user_id: Optional[str] = None,
        user_info: Optional[Dict[str, Any]] = None,
        additional_prompts: Optional[Dict[str, Any]] = None,
        include_history: bool = False,
        include_documents: bool = False
    ):
        """
        Process a chat query using the appropriate agent.
        If agent_id is provided, that specific agent will be used.
        If not, the system will select the most appropriate agent based on the query.
        
        Args:
            query: The user's query
            agent_id: Optional ID of the agent to use
            thread_id: ID of the conversation thread
            user_id: Optional ID of the user
            user_info: Optional additional information about the user
            additional_prompts: Optional additional_prompts for the agent (language, units, etc.)
            include_history: Whether to include chat history in the context
            include_documents: Whether to include document content in context
        """
        # If no agent_id provided, select the most appropriate agent
        if not agent_id:
            selected_agent = AgentManager.select_agent_for_query(query=query)
            agent_id = selected_agent["agent_id"]
            agent_name = selected_agent["name"]
            confidence = selected_agent["confidence"]
        else:
            # Check if the agent exists
            if agent_id not in AgentManager.agents:
                # Try to load from database or S3
                AgentManager.get_agent(agent_id)
            
            agent_name = AgentManager.agent_metadata[agent_id]["name"]
            confidence = 1.0  # User explicitly selected this agent
        
        # Get the agent
        agent_info = AgentManager.get_agent(agent_id)
        agent = agent_info["agent"]
        additional_query = agent_info["additional_query"]
        metadata = agent_info["metadata"]
        
        # Get current date and time
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Create context message with current date and additional info
        context_message = f"Current date: {current_date}\nCurrent time: {current_time}\n"
        
        additional_query_info = ""
        # Add additional info to context if available
        if additional_query:
            additional_query_info += "\nAdditional Information:\n"
            for key, value in additional_query.items():
                additional_query_info += f"{key}: {value}\n"
        
        # Add user information if available
        if user_id or user_info:
            context_message += "\nUser Information:\n"
            if user_id:
                context_message += f"User ID: {user_id}\n"
            
            if user_info:
                for key, value in user_info.items():
                    context_message += f"{key}: {value}\n"
        
        # Add additional_prompts if available
        if additional_prompts:
            context_message += "\nadditional_prompts:\n"
            for key, value in additional_prompts.items():
                context_message += f"{key}: {value}\n"
        
        # Add document content if available and requested
        document_refs = ""
        if include_documents and "document_refs" in metadata and metadata["document_refs"]:
            document_service = get_document_service()
            context_message += "\nReference Documents:\n"
            context_message += "\nCategory Documents:\n"
            context_message += ", ".join(metadata["document_refs"].keys())
            
            for category, doc_ids in metadata["document_refs"].items():
                if doc_ids:
                    if "*" in doc_ids:
                        documents = document_service.list_documents(category)
                    else:
                        documents = document_service.get_documents_by_ids(category, doc_ids)
                    
                    if documents:
                        document_refs += f"\n## {category.upper()} DOCUMENTS\n"
                        
                        for doc in documents:
                            document_refs += f"\n### {doc.get('title', 'Untitled Document')}\n"
                            document_refs += f"{doc.get('content', '')}\n"

        # Get database service
        db_service = get_db_service()
        
        # Create input for the agent with context
        agent_input = {
            "messages": [
                {"role": "system", "content": context_message + "\n" + document_refs},
            ]
        }

        # Include chat history if requested
        if include_history:
            # Check if conversation exists
            conversation = db_service.get_conversation(thread_id)
            
            if conversation and "messages" in conversation:
                # Get previous messages (excluding system messages)
                previous_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in conversation["messages"]
                    if msg["role"] != "system"
                ]
                
                # Add previous messages to the input
                if previous_messages:
                    agent_input["messages"].extend(previous_messages)
        
        # Add the current query
        agent_input["messages"].append({"role": "user", "content": query + "\n" + additional_query_info})
        
        # Invoke the agent with the thread_id for state persistence
        final_state = agent.invoke(
            agent_input,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract the response
        response = final_state["messages"][-1].content
        
        # Save the conversation and messages to the database
        # Check if conversation exists
        conversation = db_service.get_conversation(thread_id)
        if not conversation:
            conversation_data = {
                "id": thread_id,
                "agent_id": agent_id,
                "user_id": user_id,
                "title": f"Conversation {thread_id[:8]}"
            }
            
            # Add user_id to conversation if provided
            if user_id:
                conversation_data["user_id"] = user_id
                
            db_service.create_conversation(conversation_data)
        
        # Save context message
        db_service.create_message({
            "conversation_id": thread_id,
            "role": "system",
            "agent_id": agent_id,
            "content": context_message
        })
        
        # Save user message
        db_service.create_message({
            "conversation_id": thread_id,
            "role": "user",
            "user_id": user_id,
            "content": query
        })
        
        # Save assistant message
        db_service.create_message({
            "conversation_id": thread_id,
            "role": "assistant",
            "agent_id": agent_id,
            "content": response
        })
        
        return {
            "response": response,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "thread_id": thread_id,
            "confidence": confidence
        }

