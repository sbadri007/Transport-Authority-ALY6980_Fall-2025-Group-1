from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conversation:
    id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: Any, metadata: Dict[str, Any] = None):
        """Add a message to conversation"""
        message = Message(
            role=role,
            content=content if isinstance(content, str) else str(content),
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, n: int = 5) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'messages': [
                {
                    'role': m.role,
                    'content': m.content,
                    'timestamp': m.timestamp.isoformat(),
                    'metadata': m.metadata
                }
                for m in self.messages
            ],
            'context': self.context,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ConversationManager:
    """Manages conversation state and history"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversations: Dict[str, Conversation] = {}
        self.max_history = config['mbta_orchestrator']['conversation']['max_history']
        
        logger.info("ConversationManager initialized")
    
    async def get_or_create(
        self, 
        conversation_id: Optional[str], 
        user_id: str = "default"
    ) -> Conversation:
        """Get existing conversation or create new one"""
        
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
            logger.info(f"Retrieved conversation {conversation_id}")
            return conversation
        
        # Create new conversation
        new_id = conversation_id or self._generate_id()
        conversation = Conversation(
            id=new_id,
            user_id=user_id
        )
        self.conversations[new_id] = conversation
        
        logger.info(f"Created new conversation {new_id}")
        return conversation
    
    async def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self.conversations.get(conversation_id)
    
    async def update_context(self, conversation_id: str, context: Dict[str, Any]):
        """Update conversation context"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].context.update(context)
            self.conversations[conversation_id].updated_at = datetime.now()
    
    async def delete(self, conversation_id: str):
        """Delete conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation {conversation_id}")
    
    async def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversations"""
        now = datetime.now()
        to_delete = []
        
        for conv_id, conv in self.conversations.items():
            age = (now - conv.updated_at).total_seconds() / 3600
            if age > max_age_hours:
                to_delete.append(conv_id)
        
        for conv_id in to_delete:
            await self.delete(conv_id)
        
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old conversations")
    
    def _generate_id(self) -> str:
        """Generate unique conversation ID"""
        return f"mbta_{uuid.uuid4().hex[:12]}"
    
    async def get_all_conversations(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all conversations, optionally filtered by user"""
        conversations = self.conversations.values()
        
        if user_id:
            conversations = [c for c in conversations if c.user_id == user_id]
        
        return [c.to_dict() for c in conversations]