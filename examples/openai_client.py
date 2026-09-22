"""Optional SDK client; install openai in the calling environment."""
import os
import argparse
import json
import sys
from openai import OpenAI

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("audio")
parser.add_argument("--uttid", help="Optional audio ID for the gateway")
parser.add_argument("--audio-format", choices=["pcm", "wav"],
                    help="Gateway input format (default: wav); pcm means 16 kHz mono PCM16LE")
args = parser.parse_args()
extra_body = {"repetition_penalty": 1.0, "max_completion_tokens": 512}
if args.audio_format:
    extra_body["audio_format"] = args.audio_format
if args.uttid is not None:
    extra_body["uttid"] = args.uttid

client = OpenAI(
    base_url=os.getenv("ASR_BASE_URL", "http://127.0.0.1:8000").rstrip("/") + "/v1",
    api_key=os.getenv("ASR_API_KEY") or "EMPTY",
    timeout=7200.0,
    max_retries=0,
)
with open(args.audio, "rb") as audio:
    response = client.audio.transcriptions.with_raw_response.create(
        model="fireredasr2-llm", file=audio, response_format="json",
        temperature=0,
        extra_body=extra_body,
    )
body = response.http_response.json()
print(json.dumps(body, ensure_ascii=False, indent=2))
if body.get("errors"):
    print("Transcription failed; see errors in the response.", file=sys.stderr)
    raise SystemExit(1)
