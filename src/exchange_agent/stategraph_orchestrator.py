"""
StateGraph-based Orchestrator for MBTA Agntcy
Replaces manual orchestration with LangGraph workflow
"""
import os
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from dataclasses import dataclass
import asyncio
import httpx
from opentelemetry import trace
import logging

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    The state that flows through the StateGraph.
    Each node can read from and write to this state.
    """
    # Input
    user_message: str
    conversation_id: str
    intent: str
    confidence: float
    
    # Agent execution tracking
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agents_to_call: list[str]
    agents_called: list[str]
    
    # Results from agents
    alerts_result: dict | None
    stops_result: dict | None
    planner_result: dict | None
    
    # Final output
    final_response: str
    should_end: bool


# ============================================================================
# AGENT NODES - Each agent is a node in the graph
# ============================================================================

@dataclass
class AgentConfig:
    name: str
    url: str
    port: int


# Agent configurations
AGENTS = {
    "mbta-alerts": AgentConfig("mbta-alerts", "http://localhost", 8001),
    "mbta-stops": AgentConfig("mbta-stops", "http://localhost", 8003),
    "mbta-route-planner": AgentConfig("mbta-route-planner", "http://localhost", 8002),
}


async def call_agent_api(agent_name: str, message: str) -> dict:
    """
    Call an agent via A2A protocol
    
    Agents return: {"type": "response", "payload": {"text": "...", ...}}
    We need to extract the text from payload.
    """
    agent = AGENTS[agent_name]
    url = f"{agent.url}:{agent.port}/a2a/message"
    
    payload = {
        "type": "request",
        "payload": {
            "message": message,
            "conversation_id": "stategraph-session"
        },
        "metadata": {
            "source": "stategraph-orchestrator",
            "agent": agent_name
        }
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract the actual response text from nested structure
        # Agent returns: {"type": "response", "payload": {"text": "..."}}
        if result.get("type") == "response" and "payload" in result:
            return {
                "response": result["payload"].get("text", ""),
                "payload": result["payload"]  # Keep full payload for metadata
            }
        
        return result


# ============================================================================
# NODE FUNCTIONS - Each function is a node in the graph
# ============================================================================

async def classify_intent_node(state: AgentState) -> AgentState:
    """
    First node: Classify user intent
    """
    with tracer.start_as_current_span("classify_intent_node") as span:
        span.set_attribute("user_message", state["user_message"])
        
        # Intent classification logic
        message = state["user_message"].lower()
        
        # Check for alerts/delays
        if any(word in message for word in ["alert", "delay", "issue", "problem", "disruption", "status", "running"]):
            intent = "alerts"
            confidence = 0.9
        # Check for trip planning - expanded keywords
        elif any(word in message for word in ["how do i get", "how do i go", "i want to get", "i wanna go", 
                                                "route", "directions", "travel", "from", " to ", "take me"]):
            intent = "trip_planning"
            confidence = 0.9
        # Check for stop info
        elif any(word in message for word in ["stop", "station", "find", "near", "where is", "locate"]):
            intent = "stop_info"
            confidence = 0.85
        # Greetings and general
        elif any(word in message for word in ["hi", "hello", "hey", "thanks", "bye", "how are you"]):
            intent = "general"
            confidence = 0.9
        else:
            # Default to general for anything else (weather, off-topic, etc.)
            intent = "general"
            confidence = 0.6
        
        logger.info(f"StateGraph classified: {intent} ({confidence:.2f})")
        span.set_attribute("intent", intent)
        span.set_attribute("confidence", confidence)
        
        return {
            **state,
            "intent": intent,
            "confidence": confidence,
            "messages": [HumanMessage(content=state["user_message"])]
        }


async def alerts_agent_node(state: AgentState) -> AgentState:
    """Node: Call alerts agent"""
    with tracer.start_as_current_span("alerts_agent_node"):
        logger.info(f"Calling alerts agent for: {state['user_message']}")
        result = await call_agent_api("mbta-alerts", state["user_message"])
        
        # Extract response text
        response_text = result.get("response", "")
        
        return {
            **state,
            "alerts_result": result,
            "agents_called": state.get("agents_called", []) + ["mbta-alerts"],
            "messages": [AIMessage(content=f"Alerts: {response_text}", name="alerts-agent")]
        }


async def stops_agent_node(state: AgentState) -> AgentState:
    """Node: Call stops agent"""
    with tracer.start_as_current_span("stops_agent_node"):
        logger.info(f"Calling stops agent for: {state['user_message']}")
        result = await call_agent_api("mbta-stops", state["user_message"])
        
        # Extract response text
        response_text = result.get("response", "")
        
        return {
            **state,
            "stops_result": result,
            "agents_called": state.get("agents_called", []) + ["mbta-stops"],
            "messages": [AIMessage(content=f"Stops: {response_text}", name="stops-agent")]
        }


async def planner_agent_node(state: AgentState) -> AgentState:
    """Node: Call route planner agent - directly with original message"""
    with tracer.start_as_current_span("planner_agent_node"):
        # Use original message - planner has LLM extraction
        message = state["user_message"]
        
        logger.info(f"Calling planner agent for: {message}")
        
        result = await call_agent_api("mbta-route-planner", message)
        
        # Extract response text
        response_text = result.get("response", "")
        
        return {
            **state,
            "planner_result": result,
            "agents_called": state.get("agents_called", []) + ["mbta-route-planner"],
            "messages": [AIMessage(content=f"Route: {response_text}", name="planner-agent")]
        }


async def synthesize_response_node(state: AgentState) -> AgentState:
    """
    Final node: Synthesize all agent responses into final answer
    Handles general queries without calling agents
    """
    with tracer.start_as_current_span("synthesize_response_node"):
        # Check if this is a general/greeting query
        if state["intent"] == "general":
            # Return a friendly response for greetings
            message = state["user_message"].lower()
            
            if any(word in message for word in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
                return {
                    **state,
                    "final_response": "Hello! I'm MBTA Agntcy, your Boston transit assistant. I can help you with service alerts, stop information, and trip planning. What would you like to know?",
                    "should_end": True
                }
            elif any(word in message for word in ["how are you", "what's up", "wassup", "how's it going"]):
                return {
                    **state,
                    "final_response": "I'm doing well, thank you! I'm here to help you navigate Boston's transit system. Need help with routes, schedules, or alerts?",
                    "should_end": True
                }
            elif any(word in message for word in ["thank", "thanks", "thx"]):
                return {
                    **state,
                    "final_response": "You're welcome! Let me know if you need anything else about MBTA services.",
                    "should_end": True
                }
            elif any(word in message for word in ["bye", "goodbye", "see you", "later"]):
                return {
                    **state,
                    "final_response": "Goodbye! Safe travels on the MBTA!",
                    "should_end": True
                }
            else:
                # Off-topic or unclear query
                return {
                    **state,
                    "final_response": "I'm specialized in helping with Boston MBTA transit information. I can help you with:\n• Service alerts and delays\n• Finding stops and stations\n• Planning routes and trips\n\nWhat can I help you with today?",
                    "should_end": True
                }
        
        # Collect all agent responses for non-general queries
        responses = []
        
        if state.get("alerts_result"):
            alert_response = state["alerts_result"].get("response", "")
            if alert_response and alert_response.strip():
                responses.append(alert_response)
        
        if state.get("stops_result"):
            stop_response = state["stops_result"].get("response", "")
            # Only add if it's not an error message
            if (stop_response and 
                stop_response.strip() and 
                "couldn't retrieve" not in stop_response.lower() and
                "couldn't find" not in stop_response.lower() and
                "sorry, i couldn't find any stops matching" not in stop_response.lower() and
                "failed to fetch" not in stop_response.lower()):
                responses.append(stop_response)
        
        if state.get("planner_result"):
            planner_response = state["planner_result"].get("response", "")
            if planner_response and planner_response.strip():
                responses.append(planner_response)
        
        # Simple synthesis - join responses with clear separation
        if responses:
            final_response = "\n\n".join(filter(None, responses))
        else:
            # Fallback if no responses
            final_response = "I received your request but couldn't generate a complete response. Please try rephrasing your question or ask about MBTA service alerts."
        
        # Make sure we have something to return
        if not final_response or final_response.strip() == "":
            final_response = "I'm processing your request. Please try asking about MBTA service alerts or other transit information."
        
        return {
            **state,
            "final_response": final_response,
            "should_end": True
        }


# ============================================================================
# ROUTING FUNCTIONS - Conditional edges that decide next node
# ============================================================================

def route_after_intent(state: AgentState) -> Literal["alerts", "stops", "planner", "synthesize"]:
    """
    Conditional edge after intent classification.
    Decides which agent(s) to call based on intent.
    """
    intent = state["intent"]
    
    if intent == "alerts":
        return "alerts"
    elif intent == "stops" or intent == "stop_info":
        return "stops"
    elif intent == "trip_planning":
        # FIXED: Go directly to planner, don't call stops agent
        return "planner"  # ← Changed from "stops"
    elif intent == "general":
        # For general queries, skip agents and go straight to synthesize
        return "synthesize"
    else:
        # Unknown intent - go to synthesize
        return "synthesize"


def route_after_stops(state: AgentState) -> Literal["planner", "synthesize"]:
    """
    Conditional edge after stops node.
    If trip planning intent, go to planner. Otherwise synthesize.
    """
    if state["intent"] == "trip_planning":
        return "planner"
    else:
        return "synthesize"


def route_after_alerts(state: AgentState) -> Literal["stops", "synthesize"]:
    """
    Conditional edge after alerts node.
    For general queries, continue to stops. Otherwise synthesize.
    """
    if state["intent"] == "general":
        return "stops"
    else:
        return "synthesize"


def route_after_planner(state: AgentState) -> Literal["synthesize"]:
    """Always go to synthesis after planner"""
    return "synthesize"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def build_mbta_graph() -> StateGraph:
    """Build and compile the StateGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("alerts", alerts_agent_node)
    workflow.add_node("stops", stops_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("synthesize", synthesize_response_node)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # Add conditional edges based on intent
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "alerts": "alerts",
            "stops": "stops",
            "planner": "planner",
            "synthesize": "synthesize"
        }
    )
    
    # Routing from alerts
    workflow.add_conditional_edges(
        "alerts",
        route_after_alerts,
        {
            "stops": "stops",
            "synthesize": "synthesize"
        }
    )
    
    # Routing from stops
    workflow.add_conditional_edges(
        "stops",
        route_after_stops,
        {
            "planner": "planner",
            "synthesize": "synthesize"
        }
    )
    
    # Routing from planner
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "synthesize": "synthesize"
        }
    )
    
    # End after synthesis
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# MAIN ORCHESTRATOR FUNCTION
# ============================================================================

class StateGraphOrchestrator:
    """Main orchestrator using LangGraph StateGraph"""
    
    def __init__(self):
        self.graph = build_mbta_graph()
    
    async def process_message(self, user_message: str, conversation_id: str) -> dict:
        """
        Process a user message through the StateGraph.
        
        Args:
            user_message: The user's query
            conversation_id: Unique conversation identifier
            
        Returns:
            dict with final response and metadata
        """
        with tracer.start_as_current_span("stategraph_orchestrator") as span:
            span.set_attribute("conversation_id", conversation_id)
            
            # Initial state
            initial_state: AgentState = {
                "user_message": user_message,
                "conversation_id": conversation_id,
                "intent": "",
                "confidence": 0.0,
                "messages": [],
                "agents_to_call": [],
                "agents_called": [],
                "alerts_result": None,
                "stops_result": None,
                "planner_result": None,
                "final_response": "",
                "should_end": False
            }
            
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Extract results
            span.set_attribute("intent", final_state["intent"])
            span.set_attribute("agents_called", ",".join(final_state["agents_called"]))
            
            return {
                "response": final_state["final_response"],
                "intent": final_state["intent"],
                "confidence": final_state["confidence"],
                "agents_called": final_state["agents_called"],
                "metadata": {
                    "conversation_id": conversation_id,
                    "graph_execution": "completed"
                }
            }
    
    def visualize_graph(self, output_path: str = "graph_visualization.png"):
        """Generate a visualization of the graph structure."""
        try:
            from IPython.display import Image, display
            graph_image = self.graph.get_graph().draw_mermaid_png()
            
            with open(output_path, "wb") as f:
                f.write(graph_image)
            
            print(f"Graph visualization saved to {output_path}")
        except ImportError:
            print("Install pygraphviz for visualization: pip install pygraphviz")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Test the StateGraph orchestrator"""
    orchestrator = StateGraphOrchestrator()
    
    # Test queries
    test_queries = [
        "Are there Red Line delays?",
        "Find stops near Harvard",
        "How do I get from Park Street to MIT?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = await orchestrator.process_message(query, f"test-{hash(query)}")
        
        print(f"\nIntent: {result['intent']} (confidence: {result['confidence']})")
        print(f"Agents Called: {', '.join(result['agents_called'])}")
        print(f"\nResponse:\n{result['response']}")


if __name__ == "__main__":
    asyncio.run(main())