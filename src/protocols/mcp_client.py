from typing import Dict, Any, List, Optional
import httpx
import logging

from ..observability.otel_config import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer("mcp-client")

class MCPClient:
    """Model Context Protocol client for calling MBTA MCP tools"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)
        
        logger.info("MCPClient initialized")
    
    async def list_tools(self, service_url: str) -> List[Dict[str, Any]]:
        """List available tools from an MCP service"""
        
        with tracer.start_as_current_span("mcp_list_tools"):
            try:
                response = await self.client.post(
                    f"{service_url}/mcp/tools/list",
                    json={}
                )
                response.raise_for_status()
                
                data = response.json()
                tools = data.get('tools', [])
                
                logger.info(f"Listed {len(tools)} tools from {service_url}")
                return tools
                
            except httpx.HTTPError as e:
                logger.error(f"Error listing tools from {service_url}: {e}")
                raise
    
    async def call_tool(
        self,
        service_url: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP tool"""
        
        with tracer.start_as_current_span("mcp_call_tool") as span:
            span.set_attribute("service_url", service_url)
            span.set_attribute("tool_name", tool_name)
            
            try:
                response = await self.client.post(
                    f"{service_url}/mcp/tools/call",
                    json={
                        'name': tool_name,
                        'arguments': arguments
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                
                logger.info(f"Called MCP tool {tool_name} on {service_url}")
                return result
                
            except httpx.HTTPError as e:
                logger.error(f"Error calling MCP tool {tool_name}: {e}")
                span.record_exception(e)
                raise
    
    async def get_resource(
        self,
        service_url: str,
        resource_uri: str
    ) -> Dict[str, Any]:
        """Get an MCP resource"""
        
        with tracer.start_as_current_span("mcp_get_resource"):
            try:
                response = await self.client.post(
                    f"{service_url}/mcp/resources/read",
                    json={'uri': resource_uri}
                )
                response.raise_for_status()
                
                return response.json()
                
            except httpx.HTTPError as e:
                logger.error(f"Error getting resource {resource_uri}: {e}")
                raise
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()