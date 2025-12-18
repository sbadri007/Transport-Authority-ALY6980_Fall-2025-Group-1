from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import contextlib  # Added for None tracer handling

from .conversation_manager import ConversationManager
from .agent_executor import AgentExecutor
from .orchestrator_behavior import OrchestratorBehavior
from ..observability.otel_config import setup_otel, get_tracer, get_meter
from ..observability.metrics import MetricsCollector
from ..observability.clickhouse_logger import get_clickhouse_logger
from ..protocols.a2a_server import A2AServer
from ..protocols.mcp_client import MCPClient
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup OpenTelemetry
setup_otel("mbta-orchestrator", "http://localhost:4317")
tracer = get_tracer("mbta-orchestrator")
meter = get_meter("mbta-orchestrator")

# Initialize ClickHouse logger
ch_logger = get_clickhouse_logger()

app = FastAPI(title="MBTA Orchestration Server", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrchestrationRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}

class OrchestrationResponse(BaseModel):
    result: Dict[str, Any]
    agents_used: List[str]
    execution_flow: Dict[str, Any]
    conversation_id: str
    timestamp: str

class MBTAOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_manager = ConversationManager(config)
        self.agent_executor = AgentExecutor(config)
        self.behavior = OrchestratorBehavior()
        self.metrics = MetricsCollector(meter) if meter else None
        
        # A2A Server for agent communication
        self.a2a_server = A2AServer(config['protocols']['a2a'])
        
        # MCP Client for MBTA tools
        self.mcp_client = MCPClient(config)
        
        logger.info("MBTAOrchestrator initialized with OrchestratorBehavior")
    
    async def orchestrate(self, request: OrchestrationRequest) -> OrchestrationResponse:
        """Main orchestration logic with detailed behavior tracking"""
        
        # Fixed: Handle None tracer properly
        span_context = tracer.start_as_current_span("mbta_orchestrate") if tracer else contextlib.nullcontext()
        
        with span_context as span:
            if span:  # Only set attributes if span exists
                span.set_attribute("conversation_id", request.conversation_id or "new")
            
            try:
                # Step 1: Get or create conversation
                conversation = await self.conversation_manager.get_or_create(
                    request.conversation_id
                )
                
                # Log incoming message
                ch_logger.log_conversation(
                    conversation_id=conversation.id,
                    user_id="default",
                    role='user',
                    content=request.message,
                    intent=request.context.get('intent', 'unknown'),
                    routed_to_orchestrator=True,
                    metadata=request.context
                )
                
                # Step 2: Add user message to history
                conversation.add_message('user', request.message)
                
                # Step 3: Use behavior to select agents
                intent = request.context.get('intent', 'general')
                agents_to_call = self.behavior.select_agents(
                    intent=intent,
                    message=request.message,
                    context=request.context
                )
                
                if span:
                    span.set_attribute("agents_selected", len(agents_to_call))
                    span.set_attribute("intent", intent)
                
                # Track execution flow for transparency
                execution_flow = {
                    'intent': intent,
                    'agents_selected': [a['name'] for a in agents_to_call],
                    'execution_strategy': {
                        a['name']: a['execution_strategy'] 
                        for a in agents_to_call
                    },
                    'priorities': {
                        a['name']: a['priority']
                        for a in agents_to_call
                    }
                }
                
                logger.info(f"📋 Execution plan: {execution_flow}")
                
                # Step 4: Execute agents
                agent_results = await self.agent_executor.execute_agents(
                    agents=agents_to_call,
                    message=request.message,
                    context=request.context,
                    conversation=conversation
                )
                
                # LOG: Agent invocations
                for agent, result in zip(agents_to_call, agent_results):
                    ch_logger.log_agent_invocation(
                        invocation_id=f"inv_{uuid.uuid4().hex[:8]}",
                        conversation_id=conversation.id,
                        agent_name=agent['name'],
                        duration_ms=result.get('duration_ms', 0),
                        status=result.get('status', 'unknown'),
                        error_message=result.get('error', ''),
                        request_payload={'message': request.message},
                        response_payload=result.get('data', {})
                    )
                
                # Step 5: Use behavior to synthesize results
                final_result = self.behavior.synthesize_responses(
                    agent_responses=agent_results,
                    intent=intent
                )
                
                # Step 6: Add to conversation
                conversation.add_message('assistant', final_result)
                
                # Log assistant response
                ch_logger.log_conversation(
                    conversation_id=conversation.id,
                    user_id="default",
                    role='assistant',
                    content=str(final_result),
                    intent=intent,
                    routed_to_orchestrator=True,
                    metadata={
                        'agents_used': [a['name'] for a in agents_to_call],
                        'execution_flow': execution_flow
                    }
                )
                
                # Metrics
                if self.metrics:
                    self.metrics.record_request()
                    self.metrics.record_agent_invocations(len(agents_to_call))
                
                return OrchestrationResponse(
                    result=final_result,
                    agents_used=[a['name'] for a in agents_to_call],
                    execution_flow=execution_flow,
                    conversation_id=conversation.id,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Orchestration error: {e}", exc_info=True)
                if span:
                    span.record_exception(e)
                if self.metrics:
                    self.metrics.record_error()
                raise HTTPException(status_code=500, detail=str(e))

# Initialize orchestrator
import yaml
import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
agents_config_path = PROJECT_ROOT / 'config' / 'agents.yaml'
config_path = PROJECT_ROOT / 'config' / 'config.yaml'

with open(agents_config_path) as f:
    agents_config = yaml.safe_load(f)
with open(config_path) as f:
    config = yaml.safe_load(f)

orchestrator = MBTAOrchestrator(config)

@app.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate(request: OrchestrationRequest):
    """Main orchestration endpoint"""
    return await orchestrator.orchestrate(request)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mbta-orchestrator"}

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    conversation = await orchestrator.conversation_manager.get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation.to_dict()

@app.get("/behavior/explain")
async def explain_behavior():
    """Explain orchestrator behavior and routing rules"""
    return {
        "intent_mappings": orchestrator.behavior.intent_agent_map,
        "agent_dependencies": orchestrator.behavior.agent_dependencies,
        "execution_strategies": orchestrator.behavior.execution_strategies
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)