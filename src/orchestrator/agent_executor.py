from typing import Dict, Any, List
import asyncio
import httpx
import logging
import time
import contextlib  # Added

from ..observability.otel_config import get_tracer
from ..protocols.a2a_client import A2AClient

logger = logging.getLogger(__name__)

class AgentExecutor:
    """Executes MBTA agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.a2a_client = A2AClient(config)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Load agent configurations
        self.agents = self._load_agents()
        
        logger.info(f"AgentExecutor initialized with {len(self.agents)} agents")
    
    def _load_agents(self) -> Dict[str, Dict[str, Any]]:
        """Load agent configurations"""
        import yaml
        from pathlib import Path
        
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        agents_config_path = PROJECT_ROOT / 'config' / 'agents.yaml'
        
        with open(agents_config_path) as f:
            agents_config = yaml.safe_load(f)
        
        agents = {}
        for agent in agents_config['agents']:
            agents[agent['name']] = agent
        
        return agents
    
    async def execute_agents(
        self,
        agents: List[Dict[str, Any]],
        message: str,
        context: Dict[str, Any],
        conversation: Any
    ) -> List[Dict[str, Any]]:
        """Execute multiple agents concurrently"""
        
        # Fixed: Get tracer and handle None properly
        tracer = get_tracer("agent-executor")
        span_context = tracer.start_as_current_span("execute_agents") if tracer else contextlib.nullcontext()
        
        with span_context as span:
            if span:
                span.set_attribute("num_agents", len(agents))
        
            # Execute all agents concurrently
            tasks = [
                self._execute_single_agent(agent, message, context, conversation)
                for agent in agents
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            agent_results = []
            for agent, result in zip(agents, results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent['name']} failed: {result}")
                    agent_results.append({
                        'agent_name': agent['name'],
                        'status': 'error',
                        'error': str(result),
                        'duration_ms': 0
                    })
                else:
                    agent_results.append(result)
            
            return agent_results
    
    async def _execute_single_agent(
        self,
        agent: Dict[str, Any],
        message: str,
        context: Dict[str, Any],
        conversation: Any
    ) -> Dict[str, Any]:
        """Execute a single agent"""
        
        agent_name = agent['name']
        agent_type = agent.get('type', 'a2a')
        
        start_time = time.time()
        
        try:
            if agent_type == 'a2a':
                result = await self._execute_a2a_agent(agent, message, context)
            else:
                result = await self._execute_rest_agent(agent, message, context)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                'agent_name': agent_name,
                'status': 'success',
                'data': result,
                'duration_ms': duration_ms
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Error executing agent {agent_name}: {e}")
            return {
                'agent_name': agent_name,
                'status': 'error',
                'error': str(e),
                'duration_ms': duration_ms
            }
    
    async def _execute_a2a_agent(
        self,
        agent: Dict[str, Any],
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute A2A agent"""
        
        logger.info(f"Calling A2A agent: {agent['name']}")
        
        result = await self.a2a_client.send_message(
            agent_url=agent['service_url'],
            message=message,
            context=context
        )
        
        return result
    
    async def _execute_rest_agent(
        self,
        agent: Dict[str, Any],
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute REST agent"""
        
        logger.info(f"Calling REST agent: {agent['name']}")
        
        response = await self.http_client.post(
            f"{agent['service_url']}/query",
            json={
                'message': message,
                'context': context
            }
        )
        response.raise_for_status()
        
        return response.json()