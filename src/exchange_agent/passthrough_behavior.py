from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PassthroughBehavior:
    """Determines when to pass through to MBTA Orchestrator"""
    
    def __init__(self):
        # Intents that require orchestrator (MBTA-specific queries)
        self.orchestrator_intents = {
            'alerts',
            'trip_planning',
            'stop_info',
            'predictions',
            'schedule'
        }
        
        # Keywords that strongly indicate MBTA queries
        self.mbta_keywords = {
            'mbta', 'train', 'bus', 'subway', 'station', 'stop',
            'red line', 'green line', 'blue line', 'orange line',
            'schedule', 'arrival', 'departure', 'delay', 'alert',
            'route', 'trip', 'directions', 'travel'
        }
        
        # Greeting patterns that should NOT go to orchestrator
        self.greeting_patterns = {
            'hi', 'hello', 'hey', 'good morning', 'good afternoon',
            'good evening', 'howdy', 'greetings', 'sup', 'yo',
            'how are you', 'what\'s up', 'whats up'
        }
    
    def should_route_to_orchestrator(self, llm_response: Dict[str, Any]) -> bool:
        """Determine if request should be routed to orchestrator"""
        
        # Get the original user message (most important!)
        original_message = llm_response.get('original_message', '').lower()
        
        # Check if it's a simple greeting FIRST
        if self._is_greeting(original_message):
            logger.info(f"💬 Handling locally: Simple greeting detected: '{original_message}'")
            return False
        
        # Factor 1: LLM explicitly needs MBTA data
        if llm_response.get('needs_mbta_data', False):
            logger.info("✅ Routing to orchestrator: LLM needs MBTA data")
            return True
        
        # Factor 2: Intent is MBTA-specific
        intent = llm_response.get('intent', 'general')
        if intent in self.orchestrator_intents:
            logger.info(f"✅ Routing to orchestrator: Intent is {intent}")
            return True
        
        # Factor 3: Contains MBTA keywords
        if any(keyword in original_message for keyword in self.mbta_keywords):
            logger.info("✅ Routing to orchestrator: Contains MBTA keywords")
            return True
        
        # Factor 4: Intent is 'general' and message is short - probably casual conversation
        if intent == 'general' and len(original_message.split()) <= 5:
            logger.info(f"💬 Handling locally: Short general message")
            return False
        
        # Factor 5: Low confidence BUT check if it might be MBTA-related
        confidence = llm_response.get('confidence', 0)
        if confidence < 0.6:
            # Only route if there's some indication it's MBTA-related
            if any(keyword in original_message for keyword in self.mbta_keywords):
                logger.info(f"⚠️  Routing to orchestrator: Low confidence but has MBTA keywords")
                return True
            else:
                logger.info(f"💬 Handling locally: Low confidence but no MBTA indicators")
                return False
        
        # No MBTA-related indicators found - handle with LLM directly
        logger.info("💬 Handling locally: General conversation")
        return False


    def _is_greeting(self, message: str) -> bool:
        """Check if message is a simple greeting"""
        message_lower = message.lower().strip()
        
        # Check exact matches
        if message_lower in self.greeting_patterns:
            return True
        
        # Check if message starts with greeting
        for greeting in self.greeting_patterns:
            if message_lower.startswith(greeting):
                return True
        
        # Check if very short (1-3 words) and no MBTA keywords
        words = message_lower.split()
        if len(words) <= 3 and not any(kw in message_lower for kw in self.mbta_keywords):
            return True
        
        return False
    
    def explain_routing_decision(self, llm_response: Dict[str, Any]) -> str:
        """Explain why routing decision was made (for debugging)"""
        should_route = self.should_route_to_orchestrator(llm_response)
        
        if should_route:
            reasons = []
            if llm_response.get('needs_mbta_data'):
                reasons.append("LLM needs MBTA data")
            if llm_response.get('intent') in self.orchestrator_intents:
                reasons.append(f"Intent: {llm_response.get('intent')}")
            if llm_response.get('confidence', 1.0) < 0.6:
                reasons.append(f"Low confidence: {llm_response.get('confidence')}")
            
            return f"Routing to orchestrator because: {', '.join(reasons)}"
        else:
            return "Handling with LLM directly - general conversation"