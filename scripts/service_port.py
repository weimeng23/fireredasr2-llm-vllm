"""Share the selected public port with Docker's separate healthcheck process."""
from pathlib import Path

DEFAULT_PORT = 8000
HEALTH_PORT_FILE = Path("/tmp/fireredasr2-public-port")


def port_number(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    return port
