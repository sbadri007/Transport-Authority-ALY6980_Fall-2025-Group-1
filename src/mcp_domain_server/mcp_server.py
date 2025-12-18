# src/mcp_domain_server/mcp_server.py

"""
MCP Domain Server - Uses cubismod/mbta-mcp library's built-in server
This is a thin wrapper that launches the actual server
"""

from mbta_mcp import server
import logging
import sys

# Configure logging to stderr (stdio is for MCP communication)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

def main():
    """
    Start the MBTA MCP server
    This uses the server implementation from mbta_mcp library
    """
    logger.info("=" * 60)
    logger.info("Starting MBTA MCP Domain Server")
    logger.info("Using: cubismod/mbta-mcp library")
    logger.info("=" * 60)
    
    # Run the mbta_mcp server's main function
    # This handles all MCP protocol communication
    server.main()

if __name__ == "__main__":
    main()