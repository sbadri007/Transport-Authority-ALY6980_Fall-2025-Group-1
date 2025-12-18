from typing import Dict, Any, Callable
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class A2AMessage(BaseModel):
    type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any]

class A2AServer:
    """A2A protocol server for receiving agent messages"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.router = APIRouter(prefix="/a2a", tags=["a2a"])
        self.message_handlers: Dict[str, Callable] = {}
        
        self._setup_routes()
        
        logger.info("A2AServer initialized")
    
    def _setup_routes(self):
        """Setup A2A routes"""
        
        @self.router.post("/message")
        async def receive_message(message: A2AMessage):
            """Receive A2A message from another agent"""
            
            logger.info(f"Received A2A message of type: {message.type}")
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.type)
            
            if not handler:
                logger.warning(f"No handler for message type: {message.type}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported message type: {message.type}"
                )
            
            try:
                result = await handler(message.payload, message.metadata)
                
                return {
                    'type': 'response',
                    'payload': result,
                    'metadata': {'status': 'success'}
                }
                
            except Exception as e:
                logger.error(f"Error handling A2A message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered A2A handler for type: {message_type}")
    
    def get_router(self) -> APIRouter:
        """Get FastAPI router"""
        return self.router