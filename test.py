import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings, DatabaseType
from app.tools import AVAILABLE_TOOLS
from app.database import get_db_service
from app.services.documents.documents_manager import get_document_service

logger = logging.getLogger(__name__)

supervisor_prompt = """
You are the Supervisor Agent, responsible for managing a set of agents and directing incoming queries to the most appropriate agent based on their expertise. Your task is to ensure that the query is handled effectively by one of the available agents.

Here is a summary of your responsibilities:

Evaluate the Query: When a query is received, analyze its content to understand the subject matter, context, and intent.
Agent List: You have access to a list of agents, each with:
Prompt: The default instruction or task description for the agent.
Categories: The categories of expertise the agent specializes in (e.g., "Customer Support", "Technical Assistance", "Documentation").
Keywords: A set of keywords that are relevant to the agent's expertise (e.g., "error", "product", "installation").
Tools: The tools the agent has available to assist in responding (e.g., "Database Access", "Document Search", "Knowledge Base").
Select the Right Agent: Your goal is to choose the agent that is most qualified to handle the query. Consider the following factors:
The categories that best match the query topic.
The keywords that are most relevant to the query.
The tools the agent has, which may allow them to provide a more detailed or accurate response.
The agent prompt, ensuring the agent is best suited to handle the content of the query.

## Response
The response MUST be only agent_id or None
"""

class SupervisorAgent:
    agent_id: str = "supervisor"
    name: str = "supervisor"
    prompt: str = supervisor_prompt
    model_name: str = settings.DEFAULT_LLM_MODEL
    query: str = "Return the agent id that is most qualified to handle the query."
    agents: Dict[str, Any] = {}
    tool_names: List[str] = []
    document_refs: Dict[str, List[str]] = {}

    checkpointer = MemorySaver()

    # Initialize tools
    tools = [AVAILABLE_TOOLS[tool_name] for tool_name in (tool_names or []) if tool_name in AVAILABLE_TOOLS]
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        api_key=settings.GOOGLE_API_KEY,
    ) 
    agent_graph = create_react_agent(
        model=model,
        checkpointer=checkpointer,
        prompt=prompt,
        tools=tools
    )

    def __init__(self, agents: Dict[str, Any]):
        """
        Initializes the SupervisorAgent class with optional parameters to customize agent attributes.
        """
        self.agents = agents

    async def process_chat(
            self,
        query: str,
        thread_id: str,
        user_id: Optional[str] = None,
        user_info: Optional[Dict[str, Any]] = None,
        include_history: bool = False,
    ):
        """
        Process a query, choose the right agent based on the content, and return the response.
        """
        if not SupervisorAgent.agent_graph:
            logger.error("Agent graph is not initialized.")
            return {"response": "Error: Agent graph not initialized."}

        # If no agent_id provided, select the most appropriate agent
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Create context message
        context_message = f"Current date: {current_date}\nCurrent time: {current_time}\n"
        
        if user_id or user_info:
            context_message += "\nUser Information:\n"
            if user_id:
                context_message += f"User ID: {user_id}\n"
            if user_info:
                for key, value in user_info.items():
                    context_message += f"{key}: {value}\n"

        context_message += "\nList Agents:\n"
        context_message += "\n".join([f"{agent_id}: \n Categories: {agent['categories']} \n Keyword: {agent["keywords"]}" for agent_id, agent in self.agents.items()])
        # Get the database service
        db_service = get_db_service()

        # Create input for the agent with context
        agent_input = {
            "messages": [
                {"role": "system", "content": context_message},
            ]
        }

        if include_history:
            conversation = db_service.get_conversation(thread_id)
            if conversation and "messages" in conversation:
                previous_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in conversation["messages"]
                    if msg["role"] != "system"
                ]
                if previous_messages:
                    agent_input["messages"].extend(previous_messages)


        # Add the current query
        agent_input["messages"].append({"role": "user", "content": query})

        # Invoke the agent
        try:
            final_state = SupervisorAgent.agent_graph.invoke(agent_input, config={"configurable": {"thread_id": thread_id}})
            response = final_state["messages"][-1].content
            return {"response": response}

        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            return {"response": "Error processing the query."}

# Usage example:
from app.services.agent_manager import AgentManager

import asyncio
AgentManager.load_agents_from_db()
agents = {
        key: {k: v for k, v in value.items() if k != "prompt"}
        for key, value in AgentManager.agent_metadata.items()
    }
supervisor = SupervisorAgent(agents=agents)
response = asyncio.run(supervisor.process_chat(query = "Vorrei andare al mare", thread_id = "1234"))
print(response)
