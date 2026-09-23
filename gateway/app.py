"""File-based ASR gateway with bounded admission and local audio preprocessing."""
import asyncio
import hmac
import io
import logging
import math
import multiprocessing
import os
import socket
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .audio import AudioError, cut_wav, join_text, prepare_audio
from .scheduler import FragmentScheduler, run_in_process, run_in_thread

logger = logging.getLogger("fireredasr2.gateway")
# Inherit Uvicorn's configured handler and level so periodic INFO logs are visible.
status_logger = logging.getLogger("uvicorn.error.gateway_status")
MODEL = "fireredasr2-llm"


@dataclass(frozen=True)
class Settings:
    backend_url: str = "http://127.0.0.1:8001"
    backend_api_key: str = ""
    api_key: str = ""
    chunk_seconds: float = 30
    vad_enabled: bool = True
    vad_mode: int = 1
    vad_frame_ms: int = 20
    vad_silence_ms: int = 1500
    vad_padding_ms: int = 200
    max_upload_mb: int = 256
    max_batch_files: int = 32
    max_audio_seconds: float = 3600
    queue_capacity: int = 256
    preprocess_concurrency: int = 4
    backend_concurrency: int = 8
    backend_timeout: float = 120
    max_completion_tokens: int = 512
    media_timeout: float = 180
    status_log_interval: float = 10

    @classmethod
    def from_env(cls):
        env = os.environ
        if "MAX_ACTIVE_JOBS" in env:
            raise ValueError("MAX_ACTIVE_JOBS was replaced by QUEUE_CAPACITY and PREPROCESS_CONCURRENCY; "
                             "remove it and configure the new limits")
        return cls(
            backend_url=env.get("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/"),
            backend_api_key=env.get("BACKEND_API_KEY", ""),
            api_key=env.get("ASR_API_KEY", ""),
            chunk_seconds=float(env.get("CHUNK_SECONDS", "30")),
            vad_enabled=env.get("VAD_ENABLED", "1").lower() in ("1", "true", "yes"),
            vad_mode=int(env.get("VAD_MODE", "1")),
            vad_frame_ms=int(env.get("VAD_FRAME_MS", "20")),
            vad_silence_ms=int(env.get("VAD_SILENCE_MS", "1500")),
            vad_padding_ms=int(env.get("VAD_PADDING_MS", "200")),
            max_upload_mb=int(env.get("MAX_UPLOAD_MB", "256")),
            max_batch_files=int(env.get("MAX_BATCH_FILES", "32")),
            max_audio_seconds=float(env.get("MAX_AUDIO_SECONDS", "3600")),
            queue_capacity=int(env.get("QUEUE_CAPACITY", "256")),
            preprocess_concurrency=int(env.get("PREPROCESS_CONCURRENCY", "4")),
            backend_concurrency=int(env.get("BACKEND_CONCURRENCY", "8")),
            backend_timeout=float(env.get("BACKEND_TIMEOUT_SECONDS", "120")),
            max_completion_tokens=int(env.get("MAX_COMPLETION_TOKENS", "512")),
            media_timeout=float(env.get("MEDIA_TIMEOUT_SECONDS", "180")),
            status_log_interval=float(env.get("GATEWAY_STATUS_INTERVAL_SECONDS", "10")),
        )

    def validate(self):
        if not math.isfinite(self.status_log_interval) or self.status_log_interval < 0:
            raise ValueError("GATEWAY_STATUS_INTERVAL_SECONDS must be finite and nonnegative")
        if not 0 < self.chunk_seconds <= 30:
            raise ValueError("Chunk durations must satisfy 0 < max <= 30")
        if self.vad_mode not in range(4) or self.vad_frame_ms not in (10, 20, 30):
            raise ValueError("VAD_MODE must be 0..3; VAD_FRAME_MS must be 10, 20 or 30")
        if self.vad_silence_ms <= 0 or self.vad_padding_ms < 0:
            raise ValueError("VAD_SILENCE_MS must be positive; VAD_PADDING_MS nonnegative")
        for name in ("max_upload_mb", "max_batch_files", "max_audio_seconds", "queue_capacity", "preprocess_concurrency",
                     "backend_concurrency", "backend_timeout", "max_completion_tokens",
                     "media_timeout"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class UploadLimit:
    """Bound the entire multipart body before its parser consumes it."""
    def __init__(self, app, max_bytes, api_key=""):
        self.app = app
        self.max_bytes = max_bytes
        self.api_key = api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if self.api_key and not hmac.compare_digest(
            headers.get(b"authorization", b""), f"Bearer {self.api_key}".encode()
        ):
            return await JSONResponse({"detail": "Invalid API key"}, 401)(scope, receive, send)
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = self.max_bytes + 1
        if length > self.max_bytes:
            return await JSONResponse({"detail": "Upload too large"}, 413)(scope, receive, send)
        # Buffer first so chunked HTTP uploads have the same size limit as fixed
        # bodies. max_bytes caps this, so the request never reaches disk.
        body = io.BytesIO()
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data = message.get("body", b"")
            total += len(data)
            if total > self.max_bytes:
                return await JSONResponse({"detail": "Upload too large"}, 413)(scope, receive, send)
            body.write(data)
            if not message.get("more_body", False):
                break
        body.seek(0)
        remaining = total
        delivered_empty = False

        async def replay():
            nonlocal remaining, delivered_empty
            if total == 0 and not delivered_empty:
                delivered_empty = True
                return {"type": "http.request", "body": b"", "more_body": False}
            if remaining <= 0:
                return await receive()
            data = body.read(min(65536, remaining))
            remaining -= len(data)
            return {"type": "http.request", "body": data, "more_body": remaining > 0}

        await self.app(scope, replay, send)


def create_app(settings=None, transport=None):
    cfg = settings or Settings.from_env()
    cfg.validate()

    @asynccontextmanager
    async def lifespan(app):
        app.state.admitted_files = 0
        app.state.file_tasks = set()
        app.state.file_stages = {}
        app.state.slicing = 0
        app.state.http_inflight = 0
        app.state.queue_rejections = 0
        app.state.preprocessing = asyncio.Semaphore(cfg.preprocess_concurrency)
        # Spawn avoids inheriting the event loop, HTTP connections or thread locks.
        pool = ProcessPoolExecutor(
            max_workers=cfg.preprocess_concurrency,
            mp_context=multiprocessing.get_context("spawn"),
        )
        app.state.preprocess_pool = pool
        try:
            async with httpx.AsyncClient(
                base_url=cfg.backend_url, timeout=cfg.backend_timeout,
                headers={"Authorization": f"Bearer {cfg.backend_api_key}"} if cfg.backend_api_key else {},
                transport=transport, limits=httpx.Limits(max_connections=cfg.backend_concurrency + 2),
            ) as client:
                app.state.client = client
                async with FragmentScheduler(cfg.backend_concurrency, recognize) as scheduler:
                    app.state.scheduler = scheduler
                    reporter = (asyncio.create_task(report_status())
                                if cfg.status_log_interval > 0 else None)
                    try:
                        yield
                    finally:
                        if reporter is not None:
                            reporter.cancel()
                            await asyncio.gather(reporter, return_exceptions=True)
                        # Drain file tasks (including CPU work) before closing the backend.
                        tasks = list(app.state.file_tasks)
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await run_in_thread(lambda: pool.shutdown(wait=True, cancel_futures=True))

    app = FastAPI(title="FireRedASR2-LLM WebRTC Gateway", version="2.0.0", lifespan=lifespan)
    # The upload budget covers all files in a request, plus 1 MiB of metadata.
    app.add_middleware(UploadLimit, max_bytes=cfg.max_upload_mb * 1024 * 1024 + 1024 * 1024,
                       api_key=cfg.api_key)

    async def report_status():
        host, pid = socket.gethostname(), os.getpid()
        while True:
            due = time.monotonic() + cfg.status_log_interval
            await asyncio.sleep(cfg.status_log_interval)
            lag_ms = max(0, time.monotonic() - due) * 1000
            stages = Counter(app.state.file_stages.values())
            scheduler = app.state.scheduler
            status_logger.info(
                "gateway_status host=%s pid=%d admitted=%d/%d "
                "preprocess_wait=%d upload_read=%d preprocess_run=%d preprocess_limit=%d "
                "inference_files=%d ready_files=%d pending_segments=%d "
                "active_segments=%d/%d slicing=%d http_inflight=%d cleanup=%d "
                "queue_rejections_total=%d loop_lag_ms=%.1f",
                host, pid, app.state.admitted_files, cfg.queue_capacity,
                stages["preprocess_wait"], stages["upload_read"], stages["preprocess_run"],
                cfg.preprocess_concurrency, stages["inference"], len(scheduler.ready),
                sum(len(work.pending) for work in scheduler.ready),
                len(scheduler.running), cfg.backend_concurrency,
                app.state.slicing, app.state.http_inflight, stages["cleanup"],
                app.state.queue_rejections, lag_ms,
            )

    async def auth(authorization: Annotated[str | None, Header()] = None):
        if cfg.api_key:
            expected = f"Bearer {cfg.api_key}"
            if not hmac.compare_digest((authorization or "").encode(), expected.encode()):
                raise HTTPException(401, "Invalid API key")

    @app.get("/healthz")
    @app.get("/health", include_in_schema=False)
    async def health():
        try:
            response = await app.state.client.get("/health", timeout=3)
            response.raise_for_status()
        except httpx.HTTPError:
            return JSONResponse({"status": "unavailable", "backend": "unhealthy"}, 503)
        return {"status": "ok", "backend": "healthy"}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]}

    async def recognize(pcm, start, end):
        try:
            # The scheduler bounds both slicing and backend requests.
            app.state.slicing += 1
            try:
                payload = await run_in_thread(cut_wav, pcm, start, end)
            finally:
                app.state.slicing -= 1
            app.state.http_inflight += 1
            try:
                response = await app.state.client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("chunk.wav", payload, "audio/wav")},
                    data={"model": MODEL, "response_format": "json", "temperature": "0",
                          "repetition_penalty": "1.0",
                          "max_completion_tokens": str(cfg.max_completion_tokens)},
                )
            finally:
                app.state.http_inflight -= 1
            response.raise_for_status()
            result = response.json()
            if not isinstance(result.get("text"), str):
                raise ValueError("Missing text")
            return result["text"].strip()
        except httpx.TimeoutException as exc:
            raise HTTPException(504, "ASR backend timed out; no partial transcript returned") from exc
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            logger.warning("Backend failure: %s", type(exc).__name__)
            raise HTTPException(502, "ASR backend failed; check backend logs") from exc

    async def transcribe_file(file, audio_format, response_format, request_id):
        started = time.monotonic()
        task = asyncio.current_task()
        async with app.state.preprocessing:
            app.state.file_stages[task] = "upload_read"
            upload = bytearray()
            while data := await file.read(1024 * 1024):
                upload += data
                if len(upload) > cfg.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, "Audio file too large")
            if not upload:
                raise HTTPException(400, "Audio file is empty")
            try:
                app.state.file_stages[task] = "preprocess_run"
                duration, pcm, plan = await run_in_process(
                    app.state.preprocess_pool, prepare_audio, bytes(upload), cfg, audio_format,
                )
            except AudioError as exc:
                raise HTTPException(400, str(exc)) from exc
            del upload
        app.state.file_stages[task] = "inference"
        texts = await app.state.scheduler.submit(pcm, plan)
        segments = [
            {"id": index, "start": round(start, 3), "end": round(end, 3),
             "text": texts[index], "skipped_silence": silent}
            for index, (start, end, silent) in enumerate(plan)
        ]
        text = join_text(item["text"] for item in segments)
        elapsed = time.monotonic() - started
        logger.info("request=%s duration=%.2f elapsed=%.2f chunks=%d",
                    request_id, duration, elapsed, len(plan))
        result = {"text": text}
        if response_format == "verbose_json":
            result.update(duration=round(duration, 3), segments=segments,
                          request_id=request_id, elapsed_seconds=round(elapsed, 3),
                          rtf=round(elapsed / duration, 4),
                          timestamp_type="vad_chunk_boundaries" if cfg.vad_enabled else "chunk_boundaries",
                          vad="webrtc" if cfg.vad_enabled else "disabled")
        return result

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(auth)])
    async def transcribe(
        request: Request,
        file: Annotated[list[UploadFile], File(description="One or more files; repeat the file field")],
        uttid: Annotated[list[str] | None, Form(description="Optional unique IDs, one per file in upload order")] = None,
        audio_format: Annotated[
            Literal["pcm", "wav"],
            Form(description="Format shared by all files: 16000 Hz mono PCM16LE, raw or WAV"),
        ] = "wav",
        model: Annotated[str, Form()] = MODEL,
        response_format: Annotated[str, Form()] = "json",
        stream: Annotated[bool, Form()] = False,
    ):
        tasks = []
        disconnected = None
        batch = None
        try:
            if model != MODEL:
                raise HTTPException(400, f"model must be {MODEL}")
            if response_format not in ("json", "verbose_json"):
                raise HTTPException(400, "response_format must be json or verbose_json")
            if stream:
                raise HTTPException(400, "This gateway accepts complete files; stream is not implemented")
            if len(file) > cfg.max_batch_files:
                raise HTTPException(400, f"A request may contain at most {cfg.max_batch_files} files")
            if uttid is not None:
                if len(uttid) != len(file) or any(not value.strip() for value in uttid):
                    raise HTTPException(400, "Provide one non-empty uttid per file")
                if len(set(uttid)) != len(uttid):
                    raise HTTPException(400, "uttid values must be unique within a request")
            if sum(item.size or 0 for item in file) > cfg.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, "Total audio upload too large")

            # Admission is atomic for the entire batch; count all unfinished files.
            if app.state.admitted_files + len(file) > cfg.queue_capacity:
                app.state.queue_rejections += 1
                raise HTTPException(429, "Audio queue is full", headers={"Retry-After": "5"})
            app.state.admitted_files += len(file)
            request_id = uuid.uuid4().hex
            headers = {"X-Request-ID": request_id}
            utterance_ids = uttid if uttid is not None else [f"{request_id}-{i}" for i in range(len(file))]

            async def process_file(index, upload):
                item = {"uttid": utterance_ids[index], "index": index, "filename": upload.filename}
                try:
                    result = await transcribe_file(
                        upload, audio_format, response_format, f"{request_id}-{index}",
                    )
                    item.update(status_code=200, **result)
                except HTTPException as exc:
                    item.update(status_code=exc.status_code, error=exc.detail)
                except Exception:
                    logger.exception("request=%s file_index=%d failed", request_id, index)
                    item.update(status_code=500, error="Audio processing failed; check gateway logs")
                finally:
                    app.state.file_stages[asyncio.current_task()] = "cleanup"
                    await upload.close()
                return item

            def release_file(task):
                # Done callbacks also run for tasks cancelled before their first step.
                app.state.admitted_files -= 1
                app.state.file_tasks.discard(task)
                app.state.file_stages.pop(task, None)

            for index, upload in enumerate(file):
                task = asyncio.create_task(process_file(index, upload))
                app.state.file_stages[task] = "preprocess_wait"
                task.add_done_callback(release_file)
                app.state.file_tasks.add(task)
                tasks.append(task)
            async def wait_for_disconnect():
                while (await request.receive())["type"] != "http.disconnect":
                    pass

            disconnected = asyncio.create_task(wait_for_disconnect())
            batch = asyncio.gather(*tasks)
            done, _ = await asyncio.wait((batch, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if batch not in done:
                raise HTTPException(499, "Client disconnected")
            results = await batch
            return JSONResponse({
                "request_id": request_id,
                "results": [item for item in results if item["status_code"] == 200],
                "errors": [item for item in results if item["status_code"] != 200],
            }, headers=headers)
        finally:
            # Drain all file work before closing uploads on request cancellation.
            if disconnected is not None:
                disconnected.cancel()
                await asyncio.gather(disconnected, return_exceptions=True)
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if batch is not None:
                await asyncio.gather(batch, return_exceptions=True)
            for upload in file:
                await upload.close()

    return app


app = create_app()
