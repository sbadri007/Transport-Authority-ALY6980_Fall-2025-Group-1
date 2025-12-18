# src/exchange_agent/exchange_server.py

"""
Exchange Agent - Hybrid A2A + MCP Orchestrator
Routes queries based on confidence and complexity
"""

import sys
import os

# Load environment variables FIRST (before any other imports)
from dotenv import load_dotenv
load_dotenv()  # This loads .env from current directory or parent directories

# Initialize OpenTelemetry BEFORE other imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.observability.otel_config import setup_otel
    setup_otel("exchange-agent")
except ImportError as e:
    print(f"⚠️  Could not import observability: {e}")
    print("Continuing without telemetry...")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import time
import uuid

# Add parent directory to Python path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Try relative imports first, fall back to absolute
try:
    from .mcp_client import MCPClient
    from .intent_classifier import IntentClassifier
    from .stategraph_orchestrator import StateGraphOrchestrator
except ImportError:
    from mcp_client import MCPClient
    from intent_classifier import IntentClassifier
    from stategraph_orchestrator import StateGraphOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("=" * 60)
    logger.error("❌ OPENAI_API_KEY not found in environment!")
    logger.error("=" * 60)
    logger.error("Please ensure .env file exists in project root with:")
    logger.error("  OPENAI_API_KEY=sk-...")
    logger.error("=" * 60)
    sys.exit(1)
else:
    logger.info(f"✓ OpenAI API key loaded (ends with: ...{api_key[-4:]})")

# Global instances
mcp_client: Optional[MCPClient] = None
stategraph_orchestrator: Optional[StateGraphOrchestrator] = None
intent_classifier = None  



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    Startup: Initialize MCP client and StateGraph orchestrator
    Shutdown: Cleanup resources
    """
    global mcp_client, stategraph_orchestrator, intent_classifier 
    
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Exchange Agent with Hybrid A2A + MCP Support")
    logger.info("=" * 60)
    
    # Initialize Intent Classifier FIRST (needs OpenAI key from environment)

    try:
        intent_classifier = IntentClassifier()
        logger.info("✅ Intent Classifier initialized - Embeddings cached")
    except Exception as e:
        logger.error(f"❌ Intent Classifier initialization failed: {e}")
        logger.exception(e)
        intent_classifier = None
    
    # Initialize StateGraph Orchestrator (for A2A path)
    try:
        stategraph_orchestrator = StateGraphOrchestrator()
        logger.info("✅ StateGraph Orchestrator initialized - A2A path available")
    except Exception as e:
        logger.error(f"❌ StateGraph Orchestrator initialization failed: {e}")
        logger.exception(e)
        stategraph_orchestrator = None
    
    # Initialize MCP Client (for fast path)
    try:
        mcp_client = MCPClient()
        await mcp_client.initialize()
        logger.info("✅ MCP Client initialized - Fast path available")
    except Exception as e:
        logger.warning(f"⚠️  MCP Client initialization failed: {e}")
        logger.warning("Falling back to A2A agents only")
        mcp_client = None
    
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Exchange Agent...")
    if mcp_client:
        await mcp_client.cleanup()
    logger.info("✓ Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="MBTA Exchange Agent",
    description="Hybrid A2A + MCP Orchestrator with Confidence-Based Routing",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize intent classifier
#intent_classifier = IntentClassifier()


# Request/Response models
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    path: str  # "mcp" or "a2a"
    latency_ms: int
    intent: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "MBTA Exchange Agent",
        "version": "2.0.0",
        "architecture": "Hybrid A2A + MCP",
        "mcp_available": mcp_client is not None and mcp_client._initialized,
        "stategraph_available": stategraph_orchestrator is not None,
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with hybrid A2A + MCP support
    
    Routing Decision Logic:
    1. Classify intent and get confidence score
    2. If confidence > 0.9 AND intent is simple (alerts, stops) → MCP Fast Path
    3. Otherwise → A2A Agent Path for full reasoning
    """
    
    start_time = time.time()
    query = request.query
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"📨 Received query: {query}")
    logger.info(f"   Conversation ID: {conversation_id}")
    
    # Step 1: Classify intent and confidence
    intents, confidence_scores = intent_classifier.classify_intent(query)


    
    primary_intent = intents[0] if intents else "general"
    primary_confidence = confidence_scores.get(primary_intent, 0.0)
    
    logger.info(f"🎯 Intent: {primary_intent} | Confidence: {primary_confidence:.3f}")
    
    # Step 2: Route based on confidence and intent type
    response_text = ""
    path_taken = ""
    metadata = {}
    
    # Define simple intents that can use MCP fast path
    SIMPLE_INTENTS = ["alerts", "stops", "stop_info"]
    
    # Decision: MCP Fast Path or A2A Path?
    if (primary_confidence > 0.80 and 
        primary_intent in SIMPLE_INTENTS and 
        mcp_client and 
        mcp_client._initialized):
        
        # HIGH CONFIDENCE + SIMPLE INTENT → MCP FAST PATH
        logger.info("🚀 Routing to MCP Fast Path")
        
        try:
            response_text, metadata = await handle_mcp_path(query, primary_intent)
            path_taken = "mcp"
            
        except Exception as e:
            logger.error(f"❌ MCP path failed: {e}, falling back to A2A")
            # Fallback to A2A
            response_text, a2a_metadata = await handle_a2a_path(query, conversation_id)
            path_taken = "a2a_fallback"
            metadata = {**metadata, **a2a_metadata, "mcp_error": str(e)}
    
    else:
        # LOW CONFIDENCE or COMPLEX INTENT → A2A AGENT PATH
        reason = []
        if primary_confidence <= 0.9:
            reason.append(f"confidence={primary_confidence:.3f}")
        if primary_intent not in SIMPLE_INTENTS:
            reason.append(f"complex_intent={primary_intent}")
        if not mcp_client or not mcp_client._initialized:
            reason.append("mcp_unavailable")
        
        logger.info(f"🔄 Routing to A2A Path - Reason: {', '.join(reason)}")
        
        response_text, metadata = await handle_a2a_path(query, conversation_id)
        path_taken = "a2a"
    
    # Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)
    
    logger.info(f"✅ Response generated via {path_taken} in {latency_ms}ms")
    
    return ChatResponse(
        response=response_text,
        path=path_taken,
        latency_ms=latency_ms,
        intent=primary_intent,
        confidence=primary_confidence,
        metadata=metadata
    )


async def handle_mcp_path(query: str, intent: str) -> tuple[str, Dict[str, Any]]:
    """
    Handle query using MCP fast path
    Direct tool calls to mbta-mcp server
    
    Returns: (response_text, metadata)
    """
    
    metadata = {"tools_used": []}
    
    try:
        if intent == "alerts":
            # Extract route from query if possible
            route_id = extract_route_from_query(query)
            
            logger.info(f"Calling MCP: mbta_get_alerts(route_id={route_id})")
            alerts_data = await mcp_client.get_alerts(route_id=route_id)
            
            metadata["tools_used"].append("mbta_get_alerts")
            metadata["route_id"] = route_id
            
            # Synthesize response
            response = synthesize_alerts_response(alerts_data, route_id)
            
        elif intent in ["stops", "stop_info"]:
            # Try to extract stop name from query
            stop_query = extract_stop_name_from_query(query)
            
            if stop_query:
                logger.info(f"Calling MCP: mbta_search_stops(query={stop_query})")
                stops_data = await mcp_client.search_stops(stop_query)
                
                metadata["tools_used"].append("mbta_search_stops")
                metadata["stop_query"] = stop_query
                
                response = synthesize_stops_response(stops_data, stop_query)
            else:
                # General stop info request
                logger.info("Calling MCP: mbta_list_all_stops")
                stops_data = await mcp_client.list_all_stops()
                
                metadata["tools_used"].append("mbta_list_all_stops")
                response = synthesize_stops_list_response(stops_data)
        
        else:
            # Default fallback
            response = "I can help you with MBTA information. What would you like to know?"
            metadata["tools_used"].append("none")
        
        return response, metadata
        
    except Exception as e:
        logger.error(f"Error in MCP path: {e}", exc_info=True)
        raise


async def handle_a2a_path(query: str, conversation_id: str) -> tuple[str, Dict[str, Any]]:
    """
    Handle query using A2A agent orchestration via StateGraph
    
    Returns: (response_text, metadata)
    """
    
    if not stategraph_orchestrator:
        logger.error("StateGraph orchestrator not available")
        return ("I'm having trouble processing your request right now. Please try again.", {})
    
    try:
        # Call StateGraph orchestrator
        logger.info(f"Running StateGraph orchestration for conversation: {conversation_id}")
        
        result = await stategraph_orchestrator.process_message(query, conversation_id)
        
        # Extract response and metadata from StateGraph result
        response_text = result.get("response", "")
        
        metadata = {
            "stategraph_intent": result.get("intent"),
            "stategraph_confidence": result.get("confidence"),
            "agents_called": result.get("agents_called", []),
            "graph_execution": result.get("metadata", {}).get("graph_execution", "completed")
        }
        
        logger.info(f"StateGraph completed - Agents called: {', '.join(metadata['agents_called'])}")
        
        return response_text, metadata
        
    except Exception as e:
        logger.error(f"Error in A2A path: {e}", exc_info=True)
        return (f"I encountered an error processing your request: {str(e)}", {"error": str(e)})


# ==========================================
# Helper Functions for Query Parsing
# ==========================================

def extract_route_from_query(query: str) -> Optional[str]:
    """Extract MBTA route name from query"""
    query_lower = query.lower()
    
    route_mapping = {
        "red": "Red",
        "red line": "Red",
        "orange": "Orange",
        "orange line": "Orange",
        "blue": "Blue",
        "blue line": "Blue",
        "green": "Green-B",
        "green line": "Green-B",
        "green-b": "Green-B",
        "green-c": "Green-C",
        "green-d": "Green-D",
        "green-e": "Green-E",
    }
    
    for key, value in route_mapping.items():
        if key in query_lower:
            return value
    
    return None


def extract_stop_name_from_query(query: str) -> Optional[str]:
    """Extract stop name from query"""
    query_lower = query.lower()
    
    # Common patterns
    stop_indicators = ["station", "stop", "at", "near", "from", "to"]
    
    words = query.split()
    
    # Look for words after indicators
    for i, word in enumerate(words):
        if word.lower() in stop_indicators and i + 1 < len(words):
            # Get next 1-3 words as potential stop name
            potential_name = " ".join(words[i+1:i+4])
            return potential_name.strip('?.,!')
    
    # If no indicators found, try to extract capitalized words
    capitalized = [w for w in words if w and w[0].isupper()]
    if capitalized:
        return " ".join(capitalized[:3])
    
    return None


# ==========================================
# Response Synthesis Functions (for MCP path)
# ==========================================

def synthesize_alerts_response(alerts_data: Dict[str, Any], route_id: Optional[str]) -> str:
    """Synthesize natural language response from alerts data"""
    
    alerts = alerts_data.get('data', [])
    
    if not alerts:
        if route_id:
            return f"Good news! There are currently no service alerts for the {route_id} Line. Service is operating normally."
        else:
            return "There are currently no active service alerts. All MBTA services are operating normally."
    
    # Build response
    route_name = f"the {route_id} Line" if route_id else "MBTA services"
    response_parts = [f"There {'is' if len(alerts) == 1 else 'are'} currently {len(alerts)} active alert{'s' if len(alerts) != 1 else ''} for {route_name}:\n"]
    
    for i, alert in enumerate(alerts[:3], 1):  # Show max 3 alerts
        attrs = alert.get('attributes', {})
        header = attrs.get('header', 'Service Alert')
        description = attrs.get('description', 'No details available')
        
        # Shorten description if too long
        if len(description) > 150:
            description = description[:150] + "..."
        
        response_parts.append(f"\n{i}. {header}")
        if description and description != header:
            response_parts.append(f"   {description}")
    
    if len(alerts) > 3:
        response_parts.append(f"\n\n...and {len(alerts) - 3} more alert{'s' if len(alerts) - 3 != 1 else ''}.")
    
    return "\n".join(response_parts)


def synthesize_stops_response(stops_data: Dict[str, Any], query: str) -> str:
    """Synthesize response for stop search"""
    
    stops = stops_data.get('data', [])
    
    if not stops:
        return f"I couldn't find any stops matching '{query}'. Could you try a different name or be more specific?"
    
    if len(stops) == 1:
        stop = stops[0]
        attrs = stop.get('attributes', {})
        name = attrs.get('name', 'Unknown Stop')
        return f"I found {name}. What would you like to know about it?"
    
    # Multiple stops found
    response_parts = [f"I found {len(stops)} stops matching '{query}':\n"]
    
    for i, stop in enumerate(stops[:5], 1):  # Show max 5
        attrs = stop.get('attributes', {})
        name = attrs.get('name', 'Unknown Stop')
        response_parts.append(f"{i}. {name}")
    
    if len(stops) > 5:
        response_parts.append(f"\n...and {len(stops) - 5} more.")
    
    response_parts.append("\nWhich one are you interested in?")
    
    return "\n".join(response_parts)


def synthesize_stops_list_response(stops_data: Dict[str, Any]) -> str:
    """Synthesize response for general stops list"""
    
    stops = stops_data.get('data', [])
    
    return f"The MBTA system has {len(stops)} stops across all lines. What specific stop are you looking for?"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)