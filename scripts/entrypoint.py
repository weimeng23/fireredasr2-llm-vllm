"""One container: exec vLLM alone, or supervise vLLM and the CPU gateway."""
import argparse
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from serve import build_command, log_command
from service_port import DEFAULT_PORT, HEALTH_PORT_FILE, port_number
from env_file import read_env_file

ROOT = Path(__file__).resolve().parents[1]


def positive_seconds(name, default):
    value = float(os.getenv(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def stop_processes(processes, timeout):
    # Kill the process groups too: vLLM owns GPU worker subprocesses.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise(backend_command, gateway_command, *, gateway_env, health_url,
              startup_timeout=900, shutdown_timeout=25):
    stopping = threading.Event()
    previous = {}
    processes = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda *_: stopping.set())
    # Loopback probes must not go through a corporate HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        backend = subprocess.Popen(backend_command, cwd=ROOT, start_new_session=True)
        processes.append(backend)
        deadline = time.monotonic() + startup_timeout
        ready = False
        while not stopping.is_set():
            if backend.poll() is not None:
                raise RuntimeError(f"vLLM exited before readiness (exit {backend.returncode})")
            if time.monotonic() >= deadline:
                raise RuntimeError("vLLM startup timed out")
            try:
                with opener.open(health_url, timeout=min(2, max(0.01, deadline - time.monotonic()))) as response:
                    ready = response.status == 200
            except (OSError, urllib.error.URLError):
                pass
            if ready:
                break
            stopping.wait(0.25)
        if stopping.is_set():
            return 0
        gateway = subprocess.Popen(gateway_command, cwd=ROOT, env=gateway_env,
                                   start_new_session=True)
        processes.append(gateway)
        print("vLLM ready; starting WebRTC VAD gateway", flush=True)
        log_command(gateway_command)
        while not stopping.wait(0.25):
            for name, process in (("vLLM", backend), ("gateway", gateway)):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited unexpectedly (exit {process.returncode})")
        return 0
    finally:
        stop_processes(processes, shutdown_timeout)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("vllm", "all"))
    parser.add_argument("--env-file", type=Path,
                        help="Container-local KEY=VALUE file; overrides existing environment variables")
    parser.add_argument("--port", type=port_number,
                        help="Public listening port (overrides SERVICE_PORT; default: 8000)")
    args = parser.parse_args()
    if args.env_file is not None:
        os.environ.update(read_env_file(args.env_file))
    mode = args.mode or os.getenv("SERVICE_MODE", "vllm")
    port = args.port if args.port is not None else port_number(os.getenv("SERVICE_PORT", str(DEFAULT_PORT)))
    if mode not in ("vllm", "all"):
        raise ValueError("SERVICE_MODE must be vllm or all")
    if mode == "all" and port == 8001:
        parser.error("Port 8001 is reserved for the internal vLLM backend in all mode")
    startup = positive_seconds("STARTUP_TIMEOUT_SECONDS", "900")
    shutdown = positive_seconds("SHUTDOWN_TIMEOUT_SECONDS", "25")
    if mode == "vllm":
        # ASR_API_KEY is the public API key in either mode.
        if not os.getenv("BACKEND_API_KEY") and os.getenv("ASR_API_KEY"):
            os.environ["BACKEND_API_KEY"] = os.environ["ASR_API_KEY"]
        command = build_command(port=port)
        HEALTH_PORT_FILE.write_text(str(port), encoding="ascii")
        log_command(command)
        os.execvp(command[0], command)
    backend = build_command(host="127.0.0.1", port=8001)
    HEALTH_PORT_FILE.write_text(str(port), encoding="ascii")
    gateway_env = dict(os.environ, BACKEND_URL="http://127.0.0.1:8001")
    gateway_python = os.getenv("GATEWAY_PYTHON", "/opt/gateway-venv/bin/python")
    gateway = [gateway_python, "-m", "uvicorn", "gateway.app:app",
               "--host", "0.0.0.0", "--port", str(port), "--workers", "1",
               "--timeout-graceful-shutdown", str(max(1, int(shutdown) - 2))]
    log_command(backend)
    return supervise(backend, gateway, gateway_env=gateway_env,
                     health_url="http://127.0.0.1:8001/health",
                     startup_timeout=startup, shutdown_timeout=shutdown)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Service startup failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
