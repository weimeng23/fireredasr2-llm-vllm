"""Validate a local model and exec a fixed, single-GPU vLLM command."""
import json
import os
from pathlib import Path


def build_command(host="0.0.0.0", port=8000):
    root = Path(os.getenv("MODEL_PATH", "/models/fireredasr2"))
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise SystemExit(f"Missing {root / name}; download the full converted model.")
    config = json.loads((root / "config.json").read_text())
    if "FireRedASR2ForConditionalGeneration" not in config.get("architectures", []):
        raise SystemExit("Wrong model architecture: use allendou/FireRedASR2-LLM-vllm.")
    if not list(root.glob("*.safetensors")):
        raise SystemExit("No safetensors weights found.")
    cmd = [
        "vllm", "serve", str(root),
        "--served-model-name", "fireredasr2-llm",
        "--host", host, "--port", str(port),
        "--tensor-parallel-size", "1",
        "--dtype", os.getenv("DTYPE", "bfloat16"),
        "--gpu-memory-utilization", os.getenv("GPU_MEMORY_UTILIZATION", "0.85"),
        "--max-model-len", os.getenv("MAX_MODEL_LEN", "4096"),
        "--max-num-seqs", os.getenv("MAX_NUM_SEQS", "8"),
        "--max-num-batched-tokens", os.getenv("MAX_NUM_BATCHED_TOKENS", "4096"),
        "--max-num-queued-reqs", os.getenv("MAX_NUM_QUEUED_REQS", "64"),
        "--mm-processor-cache-gb", os.getenv("MM_PROCESSOR_CACHE_GB", "0"),
        "--limit-mm-per-prompt", '{"audio":1}',
    ]
    if os.getenv("ENFORCE_EAGER", "0").lower() in ("1", "true", "yes"):
        cmd.append("--enforce-eager")
    if os.getenv("BACKEND_API_KEY"):
        cmd.extend(["--api-key", os.environ["BACKEND_API_KEY"]])
    return cmd


def log_command(command):
    visible = list(command)
    if "--api-key" in visible:
        visible[visible.index("--api-key") + 1] = "<redacted>"
    print("Starting:", " ".join(visible), flush=True)


if __name__ == "__main__":
    command = build_command()
    log_command(command)
    os.execvp(command[0], command)
