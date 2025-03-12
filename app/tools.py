from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import requests
import json
import logging
import math
from langchain_core.tools import tool
from app.config import settings

logger = logging.getLogger(__name__)

# Search Tool
@tool
def search(query: str) -> str:
    """Search the web for information."""
    if not query:
        return "Error: No search query provided."
    
    try:
        # Use Serper API for search
        headers = {
            "X-API-KEY": settings.SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        data = {
            "q": query
        }
        
        response = requests.post(
            "https://google.serper.dev/search", 
            headers=headers, 
            json=data
        )
        
        if response.status_code == 200:
            results = response.json()
            
            # Format the results
            formatted_results = "Search Results:\n"
            
            # Add organic results
            if "organic" in results:
                for i, result in enumerate(results["organic"][:3], 1):
                    formatted_results += f"{i}. {result.get('title', 'No Title')}\n"
                    formatted_results += f"   {result.get('snippet', 'No Snippet')}\n"
                    formatted_results += f"   URL: {result.get('link', 'No Link')}\n\n"
            
            return formatted_results
        else:
            return f"Error: Search API returned status code {response.status_code}"
            
    except Exception as e:
        logger.error(f"Search tool error: {str(e)}")
        return f"Error performing search: {str(e)}"

# Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    if not expression:
        return "Error: No expression provided."
    
    try:
        # Create a safe evaluation environment
        safe_globals = {"__builtins__": None}
        safe_locals = {
            "abs": abs, "round": round,
            "min": min, "max": max,
            "sum": sum, "pow": pow,
            "int": int, "float": float,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e
        }
        
        # Evaluate the expression
        result = eval(expression, safe_globals, safe_locals)
        return f"Result: {result}"
    except Exception as e:
        logger.error(f"Calculator tool error: {str(e)}")
        return f"Error calculating expression: {str(e)}"

# Database Lookup Tool
@tool
def database_lookup(entity: str) -> str:
    """Look up information in the database."""
    # This is a mock database lookup tool
    if not entity:
        return "Error: No entity specified for lookup."
    
    # Mock database with predefined entities
    mock_db = {
        "weather_api": "API for accessing weather data. Endpoint: /api/weather?location={location}",
        "user_profile": "User profile schema includes: id, name, email, created_at",
        "products": "Product catalog with categories: electronics, clothing, home goods",
        "pricing": "Pricing tiers: Basic ($10/mo), Pro ($25/mo), Enterprise ($100/mo)"
    }
    
    if entity.lower() in mock_db:
        return f"Database lookup result for '{entity}':\n{mock_db[entity.lower()]}"
    else:
        return f"No information found in database for '{entity}'."

# Dictionary of available tools
AVAILABLE_TOOLS = {
    "search": search,
    "calculator": calculator,
    "database_lookup": database_lookup
}
