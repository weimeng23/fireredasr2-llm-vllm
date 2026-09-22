#!/usr/bin/env python3
"""Standard-library client: python3 examples/client.py AUDIO --format verbose_json."""
import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default=os.getenv("ASR_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--format", choices=["json", "verbose_json", "text"], default="json")
    parser.add_argument("--audio-format", choices=["pcm", "wav"],
                        help="Gateway input format (default: wav); pcm means 16 kHz mono PCM16LE")
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    boundary = uuid.uuid4().hex
    body = bytearray()
    fields = {"model": "fireredasr2-llm", "response_format": args.format,
              "temperature": "0", "repetition_penalty": "1.0",
              "max_completion_tokens": "512"}
    if args.audio_format:
        fields["audio_format"] = args.audio_format
    for key, value in fields.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    media = mimetypes.guess_type(args.audio.name)[0] or "application/octet-stream"
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio"\r\nContent-Type: {media}\r\n\r\n'.encode())
    body.extend(args.audio.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    key = os.getenv("ASR_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(args.url.rstrip("/") + "/v1/audio/transcriptions",
                                     data=bytes(body), headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = response.read().decode()
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()}", file=sys.stderr)
        raise SystemExit(1)
    if args.format != "text":
        result = json.dumps(json.loads(result), ensure_ascii=False, indent=2)
    print(result)
    print(f"Elapsed: {time.monotonic() - started:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
