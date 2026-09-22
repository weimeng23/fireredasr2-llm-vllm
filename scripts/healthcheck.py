"""Identical public health endpoint in both container modes."""
import urllib.request

from service_port import DEFAULT_PORT, HEALTH_PORT_FILE, port_number


def main():
    port = port_number(HEALTH_PORT_FILE.read_text(encoding="ascii")) if HEALTH_PORT_FILE.exists() else DEFAULT_PORT
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/health", timeout=3) as response:
        if response.status != 200:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
