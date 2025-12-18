from typing import Dict, Any, List
import httpx
import logging
import contextlib  # Added

from ..observability.otel_config import get_tracer

logger = logging.getLogger(__name__)

class A2AClient:
    """Agent-to-Agent protocol client"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transport = config['protocols']['a2a']['transport']
        self.client = httpx.AsyncClient(timeout=30.0)
        
        logger.info(f"A2AClient initialized with transport: {self.transport}")
    
    async def send_message(
        self,
        agent_url: str,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send A2A message to an agent"""
        
        # Fixed: Get tracer and handle None
        tracer = get_tracer("a2a-client")
        span_context = tracer.start_as_current_span("a2a_send_message") if tracer else contextlib.nullcontext()
        
        with span_context as span:
            if span:
                span.set_attribute("agent_url", agent_url)
                span.set_attribute("transport", self.transport)
        
            try:
                # Construct A2A message
                a2a_message = {
                    'type': 'request',
                    'payload': {
                        'message': message,
                        'context': context
                    },
                    'metadata': {
                        'transport': self.transport,
                        'version': '1.0'
                    }
                }
                
                # Send via HTTP (SLIM transport)
                response = await self.client.post(
                    f"{agent_url}/a2a/message",
                    json=a2a_message
                )
                response.raise_for_status()
                
                result = response.json()
                
                logger.info(f"Sent A2A message to {agent_url}")
                return result.get('payload', {})
                
            except httpx.HTTPError as e:
                logger.error(f"Error sending A2A message to {agent_url}: {e}")
                if span:
                    span.record_exception(e)
                raise
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()