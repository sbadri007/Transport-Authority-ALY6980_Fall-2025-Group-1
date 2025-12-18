from typing import Dict, Any
import httpx
import logging

logger = logging.getLogger(__name__)

class RESTAdapter:
    """REST API adapter for legacy MBTA services"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def call_endpoint(
        self,
        url: str,
        method: str = "POST",
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Call REST endpoint"""
        
        try:
            if method.upper() == "POST":
                response = await self.client.post(url, json=data, headers=headers)
            elif method.upper() == "GET":
                response = await self.client.get(url, params=data, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"REST call failed: {e}")
            raise
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()