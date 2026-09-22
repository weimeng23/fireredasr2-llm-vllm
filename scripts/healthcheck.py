"""Identical public health endpoint in both container modes."""
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("http://127.0.0.1:8000/health", timeout=3) as response:
    if response.status != 200:
        raise SystemExit(1)
