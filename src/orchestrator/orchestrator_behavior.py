from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class OrchestratorBehavior:
    """
    Orchestrator Behavior for MBTA Server
    
    This is the "brain" of the MBTA orchestrator that decides:
    1. Which agents should handle this query?
    2. Should agents run in parallel or sequence?
    3. How should we prioritize agents?
    4. How should we combine agent responses?
    
    This implements a graph-based decision flow similar to LangGraph.
    """
    
    def __init__(self):
        """
        Initialize orchestrator behavior with routing rules
        """

        # Intent → Agent mapping
        # Maps each intent to the agents that can handle it
        self.intent_agent_map = {
            'alerts': {
                'primary': ['mbta-alerts'],
                'secondary': []
            },
            'trip_planning': {
                'primary': ['mbta-route-planner'],
                'secondary': ['mbta-stops']  # Removed mbta-predictions
            },
            'stop_info': {
                'primary': ['mbta-stops'],
                'secondary': []  # Removed mbta-predictions
            },
            'predictions': {
                'primary': ['mbta-stops'],  # Use stops agent for predictions
                'secondary': []
            },
            'schedule': {
                'primary': ['mbta-stops'],  # Use stops agent for schedules
                'secondary': []
            },
            'general': {
                'primary': ['mbta-alerts', 'mbta-stops', 'mbta-route-planner'],  # All 3 agents!
                'secondary': []
            }
        }
        
        # Agent dependencies
        # Some agents need data from other agents
        self.agent_dependencies = {
            'mbta-route-planner': ['mbta-stops'],  # Needs stop info first
        }
        
        # Execution strategies
        self.execution_strategies = {
            'parallel': ['mbta-alerts', 'mbta-predictions'],  # Can run together
            'sequential': ['mbta-route-planner']  # Needs results from previous
        }
    
    def select_agents(
        self, 
        intent: str, 
        message: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Select which agents to invoke based on intent and context
        
        This is the core routing logic - like a graph traversal in LangGraph
        
        Args:
            intent: User intent (alerts, trip_planning, etc.)
            message: Raw user message
            context: Additional context from LLM
            
        Returns:
            List of agent configurations to invoke
        """
        
        # Get agent mapping for this intent
        agent_mapping = self.intent_agent_map.get(
            intent, 
            self.intent_agent_map['general']
        )
        
        # Start with primary agents (required)
        selected_agent_names = agent_mapping['primary'].copy()
        
        # Add secondary agents based on context
        should_add_secondary = self._should_add_secondary_agents(
            intent, 
            message, 
            context
        )
        
        if should_add_secondary:
            selected_agent_names.extend(agent_mapping['secondary'])
            logger.info(f"➕ Adding secondary agents for enhanced results")
        
        # Resolve dependencies
        selected_agent_names = self._resolve_dependencies(selected_agent_names)
        
        # Load agent configurations
        agents = self._load_agent_configs(selected_agent_names)
        
        # Add execution metadata
        for agent in agents:
            agent['execution_strategy'] = self._get_execution_strategy(agent['name'])
            agent['priority'] = self._get_agent_priority(agent['name'], intent)
        
        # Sort by priority
        agents.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(
            f"🎯 Selected {len(agents)} agents: "
            f"{[a['name'] for a in agents]}"
        )
        
        return agents
    
    def _should_add_secondary_agents(
        self,
        intent: str,
        message: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Decide if we should add secondary agents for enhanced results
        
        Secondary agents provide additional context but aren't strictly required
        """
        
        # Add secondary for complex queries
        if len(message.split()) > 10:
            return True
        
        # Add secondary if LLM has high confidence we need more data
        if context.get('confidence', 0) > 0.8:
            return True
        
        # Add secondary for trip planning (always want real-time data)
        if intent == 'trip_planning':
            return True
        
        return False
    
    def _resolve_dependencies(self, agent_names: List[str]) -> List[str]:
        """
        Resolve agent dependencies
        
        If agent A depends on agent B, ensure B is called first
        """
        resolved = agent_names.copy()
        
        for agent in agent_names:
            if agent in self.agent_dependencies:
                dependencies = self.agent_dependencies[agent]
                for dep in dependencies:
                    if dep not in resolved:
                        resolved.insert(0, dep)  # Add dependency first
                        logger.info(f"🔗 Added dependency: {dep} for {agent}")
        
        return resolved
    
    def _get_execution_strategy(self, agent_name: str) -> str:
        """
        Determine execution strategy for agent (parallel vs sequential)
        """
        if agent_name in self.execution_strategies['parallel']:
            return 'parallel'
        elif agent_name in self.execution_strategies['sequential']:
            return 'sequential'
        return 'parallel'  # Default to parallel
    
    def _get_agent_priority(self, agent_name: str, intent: str) -> int:
        """
        Get priority for agent execution (higher = earlier)
        """
        priority_map = {
            'alerts': {
                'mbta-alerts': 100
            },
            'trip_planning': {
                'mbta-stops': 90,
                'mbta-route-planner': 80,
                'mbta-predictions': 70
            },
            'stop_info': {
                'mbta-stops': 100,
                'mbta-predictions': 70
            }
        }
        
        intent_priorities = priority_map.get(intent, {})
        return intent_priorities.get(agent_name, 50)  # Default priority
    
    def _load_agent_configs(self, agent_names: List[str]) -> List[Dict[str, Any]]:
        """Load full agent configurations from agents.yaml"""
        import yaml
        from pathlib import Path
        
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        agents_config_path = PROJECT_ROOT / 'config' / 'agents.yaml'
        
        with open(agents_config_path) as f:
            agents_config = yaml.safe_load(f)
        
        # Debug logging
        all_agent_names = [agent['name'] for agent in agents_config['agents']]
        logger.info(f"🔍 Available agents in config: {all_agent_names}")
        logger.info(f"🔍 Requested agent names: {agent_names}")
        
        matched_agents = [
            agent for agent in agents_config['agents']
            if agent['name'] in agent_names
        ]
        
        logger.info(f"🔍 Matched agents: {[a['name'] for a in matched_agents]}")
        
        return matched_agents
    
    def synthesize_responses(
        self,
        agent_responses: List[Dict[str, Any]],
        intent: str
    ) -> Dict[str, Any]:
        """
        Combine multiple agent responses into coherent result
        
        This is like the "reducer" step in a graph workflow
        """
        
        # Different synthesis strategies based on intent
        if intent == 'trip_planning':
            return self._synthesize_trip_planning(agent_responses)
        elif intent == 'alerts':
            return self._synthesize_alerts(agent_responses)
        elif intent == 'stop_info':
            return self._synthesize_stop_info(agent_responses)
        else:
            return self._synthesize_general(agent_responses)
    
    def _synthesize_trip_planning(
        self, 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize trip planning responses
        
        Combines route planning + real-time predictions + stop info
        """
        result = {
            'type': 'trip_plan',
            'route': None,
            'real_time_info': None,
            'stops': []
        }
        
        for response in responses:
            agent_name = response.get('agent_name')
            
            if agent_name == 'mbta-route-planner':
                result['route'] = response.get('data', {})
            elif agent_name == 'mbta-predictions':
                result['real_time_info'] = response.get('data', {})
            elif agent_name == 'mbta-stops':
                result['stops'] = response.get('data', {})
        
        return result
    
    def _synthesize_alerts(
        self,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize alert responses"""
        for response in responses:
            if response.get('agent_name') == 'mbta-alerts':
                return response.get('data', {})
        return {}
    
    def _synthesize_stop_info(
        self,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize stop info with real-time data"""
        result = {
            'stop_details': None,
            'predictions': None
        }
        
        for response in responses:
            agent_name = response.get('agent_name')
            
            if agent_name == 'mbta-stops':
                result['stop_details'] = response.get('data', {})
            elif agent_name == 'mbta-predictions':
                result['predictions'] = response.get('data', {})
        
        return result
    
    def _synthesize_general(
        self,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """General synthesis - combine all responses"""
        return {
            'results': [r.get('data', {}) for r in responses]
        }