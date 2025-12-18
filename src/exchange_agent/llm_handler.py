# src/exchange_agent/llm_handler.py

import openai
import os
import logging
from typing import Dict, Any, Optional
from .intent_classifier import create_intent_classifier

logger = logging.getLogger(__name__)

class LLMHandler:
    """
    Handles LLM interactions for the exchange agent.
    Uses OpenAI GPT-4o for response synthesis.
    Uses embedding-based intent classification (no LLM calls for intent!).
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize LLM handler with model and intent classifier.
        
        Args:
            model: OpenAI model to use for synthesis (default: gpt-4o)
        """
        self.model = model
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize embedding-based intent classifier
        self.intent_classifier = create_intent_classifier(api_key=api_key)
        
        logger.info(f"LLMHandler initialized with model: {self.model}")
        logger.info("Using embedding-based intent classification (no LLM calls for intent!)")
    
    async def process(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Process a user message with intent classification and response generation.
        
        Args:
            message: User's input message
            context: Optional context from previous interactions
            conversation_history: Optional conversation history
            
        Returns:
            Dict containing response, intent, confidence, and other metadata
        """
        try:
            # ✨ Use embedding-based intent classification (no LLM call!)
            intents, confidences = self.intent_classifier.classify_intent(message)
            
            # Determine if MBTA data is needed based on detected intents
            transit_intents = ["alerts", "trip_planning", "stop_info", "schedule"]
            needs_mbta_data = any(intent in transit_intents for intent in intents)
            
            # Get primary intent and its confidence
            primary_intent = intents[0] if intents else "general"
            primary_confidence = confidences.get(primary_intent, 0.5)
            
            # Build complete intent result with ALL possible fields
            intent_result = {
                # Core fields
                "intent": primary_intent,  # Single primary intent (for backward compat)
                "intents": intents,  # List of all detected intents
                "confidence": confidences,  # Dict of all confidence scores
                "needs_mbta_data": needs_mbta_data,
                
                # Additional fields for logging/compatibility
                "primary_intent": primary_intent,
                "primary_confidence": primary_confidence,
                "all_scores": confidences,
                
                # Summary
                "summary": f"{primary_intent} ({primary_confidence:.2f})" + 
                           (f" + {len(intents)-1} more" if len(intents) > 1 else "")
            }
            
            # For backward compatibility, expose individual variables
            intent = primary_intent
            confidence = primary_confidence
            
            # Log intent classification results
            logger.info(
                f"Intent Classification: {intent} "
                f"(confidence: {confidence:.2f}) "
                f"All intents: {intents} "
                f"Scores: {confidences}"
            )
            
            # If it's a simple greeting and high confidence, return pass-through
            if intent == "general" and confidence > 0.8:
                return {
                    "response": self._generate_greeting_response(message),
                    "intent": intent,
                    "intents": intents,
                    "confidence": confidence,
                    "needs_mbta_data": False,
                    "pass_through": True
                }
            
            # If MBTA data is needed, return intent info for orchestration
            if needs_mbta_data:
                return {
                    "response": None,  # Will be filled by orchestrator after agent calls
                    "intent": intent,
                    "intents": intents,
                    "confidence": confidence,
                    "needs_mbta_data": True,
                    "pass_through": False,
                    "intent_result": intent_result
                }
            
            # For other general queries, generate response directly
            return {
                "response": self._generate_general_response(message, context),
                "intent": intent,
                "intents": intents,
                "confidence": confidence,
                "needs_mbta_data": False,
                "pass_through": False
            }
            
        except Exception as e:
            logger.error(f"Intent processing failed: {e}", exc_info=True)
            # Fallback to safe defaults
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "intent": "general",
                "intents": ["general"],
                "confidence": 0.0,
                "needs_mbta_data": False,
                "error": str(e)
            }
    
    def _generate_greeting_response(self, message: str) -> str:
        """Generate a simple greeting response without LLM call."""
        message_lower = message.lower().strip()
        
        greetings = {
            "hello": "Hello! I'm your MBTA assistant. I can help you with service alerts, trip planning, stop information, and schedules. What would you like to know?",
            "hi": "Hi there! How can I help you with the MBTA today?",
            "hey": "Hey! What can I help you with regarding Boston's transit system?",
            "good morning": "Good morning! Ready to help you navigate the MBTA. What do you need?",
            "good afternoon": "Good afternoon! How can I assist with your transit needs?",
            "good evening": "Good evening! What MBTA information can I provide?",
            "thanks": "You're welcome! Let me know if you need anything else.",
            "thank you": "You're very welcome! Happy to help.",
            "bye": "Goodbye! Safe travels on the MBTA!",
            "goodbye": "Take care! Feel free to come back if you need more transit information."
        }
        
        for key, response in greetings.items():
            if key in message_lower:
                return response
        
        # Default greeting
        return "Hello! I'm your MBTA assistant. How can I help you today?"
    
    def _generate_general_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a general response using GPT-4o."""
        try:
            system_prompt = """You are a helpful MBTA (Massachusetts Bay Transportation Authority) assistant.
You help users with Boston's public transit system including subway, bus, commuter rail, and ferry services.
Be concise, friendly, and informative."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            # Add context if available
            if context and context.get("agent_responses"):
                context_str = "\n\nContext from transit data:\n"
                for agent, response in context["agent_responses"].items():
                    context_str += f"\n{agent}: {response}\n"
                messages.append({"role": "assistant", "content": context_str})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I apologize, but I'm having trouble generating a response. Please try again."
    
    async def synthesize_response(
        self,
        user_query: str,
        agent_responses: Dict[str, Any],
        intents: list
    ) -> str:
        """
        Synthesize a final response from multiple agent responses.
        
        Args:
            user_query: Original user query
            agent_responses: Dict of responses from different agents
            intents: List of detected intents
            
        Returns:
            Synthesized response string
        """
        try:
            # Build context from agent responses
            context_parts = []
            
            if "alerts" in agent_responses:
                context_parts.append(f"Service Alerts: {agent_responses['alerts']}")
            
            if "stops" in agent_responses:
                context_parts.append(f"Stop Information: {agent_responses['stops']}")
            
            if "planner" in agent_responses:
                context_parts.append(f"Route Planning: {agent_responses['planner']}")
            
            context_str = "\n\n".join(context_parts)
            
            system_prompt = """You are an MBTA assistant. Synthesize the information from various transit data sources into a clear, helpful response.
Be concise and focus on what's most relevant to the user's query.
Format the response in a user-friendly way."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User asked: {user_query}\n\nAvailable information:\n{context_str}\n\nProvide a helpful response:"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}", exc_info=True)
            # Return raw agent responses as fallback
            return "\n\n".join([f"{k}: {v}" for k, v in agent_responses.items()])