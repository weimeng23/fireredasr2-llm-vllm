import asyncio
import io
import struct
import wave
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.app import Settings, create_app


def audio(number=1, audio_format="wav", rate=16000, channels=1):
    pcm = struct.pack("<h", number) * 800
    if audio_format == "pcm":
        return pcm
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(2)
        dst.setframerate(rate)
        dst.writeframes(pcm)
    return out.getvalue()


def files(count=1, audio_format="wav"):
    return [("file", (f"{i}.{audio_format}", audio(i, audio_format), "application/octet-stream"))
            for i in range(1, count + 1)]


def backend_number(request):
    assert request.url.path == "/v1/audio/transcriptions"
    message = BytesParser(policy=default).parsebytes(
        f'Content-Type: {request.headers["content-type"]}\r\n\r\n'.encode() + request.content
    )
    parts = {part.get_param("name", header="content-disposition"): part
             for part in message.iter_parts()}
    assert "audio_format" not in parts
    with wave.open(io.BytesIO(parts["file"].get_payload(decode=True)), "rb") as src:
        assert (src.getframerate(), src.getnchannels(), src.getsampwidth()) == (16000, 1, 2)
        return struct.unpack("<h", src.readframes(1))[0]


def backend(request):
    return httpx.Response(200, json={"text": f"text {backend_number(request)}"})


@pytest.mark.parametrize("audio_format", [None, "wav", "pcm"])
@pytest.mark.parametrize("response_format", ["json", "verbose_json"])
def test_single_file_uses_unified_response(audio_format, response_format):
    app = create_app(Settings(vad_enabled=False), httpx.MockTransport(backend))
    with TestClient(app) as client:
        data = {"response_format": response_format}
        if audio_format:
            data["audio_format"] = audio_format
        response = client.post("/v1/audio/transcriptions", data=data,
                               files=files(audio_format=audio_format or "wav"))
        assert response.status_code == 200, response.text
        assert response.headers["x-request-id"]
        body = response.json()
        assert set(body) == {"request_id", "results", "errors"}
        assert body["errors"] == []
        assert body["request_id"] == response.headers["x-request-id"]
        assert len(body["results"]) == 1
        item = body["results"][0]
        assert item["text"] == "text 1"
        assert item["uttid"] == f'{body["request_id"]}-0'
        assert item["index"] == 0 and item["status_code"] == 200
        if response_format == "verbose_json":
            assert item["duration"] == 0.05
            assert item["request_id"] == f'{body["request_id"]}-0'
        assert app.state.active_jobs == 0


@pytest.mark.parametrize("audio_format", ["wav", "pcm"])
@pytest.mark.parametrize("response_format", ["json", "verbose_json"])
def test_batch_parallelism_and_upload_order(audio_format, response_format):
    async def run():
        second_finished = asyncio.Event()
        order = []

        async def reordered_backend(request):
            number = backend_number(request)
            if number == 1:
                await asyncio.wait_for(second_finished.wait(), timeout=2)
            order.append(number)
            if number == 2:
                second_finished.set()
            return httpx.Response(200, json={"text": f"text {number}"})

        app = create_app(Settings(vad_enabled=False), httpx.MockTransport(reordered_backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                response = await client.post(
                    "/v1/audio/transcriptions", files=files(2, audio_format),
                    data={"audio_format": audio_format, "response_format": response_format},
                )
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["request_id"] == response.headers["x-request-id"]
                assert order == [2, 1]
                assert [r["text"] for r in body["results"]] == ["text 1", "text 2"]
                assert [r["index"] for r in body["results"]] == [0, 1]
                assert [r["filename"] for r in body["results"]] == [f"1.{audio_format}", f"2.{audio_format}"]
                assert all(r["status_code"] == 200 for r in body["results"])
                assert body["errors"] == []
                if response_format == "verbose_json":
                    assert all(r["duration"] == 0.05 and len(r["segments"]) == 1 for r in body["results"])
                    assert len({r["request_id"] for r in body["results"]}) == 2
                assert app.state.active_jobs == 0

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["invalid_audio", "backend_error", "timeout", "internal_error"])
def test_batch_partial_failure_does_not_discard_success(failure):
    def fail_one(request):
        if backend_number(request) == 2:
            if failure == "backend_error":
                return httpx.Response(503)
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            if failure == "internal_error":
                raise RuntimeError("private internal detail")
        return backend(request)

    uploads = files(3)
    if failure == "invalid_audio":
        uploads[1] = ("file", ("bad.wav", b"not a wav"))
    app = create_app(Settings(vad_enabled=False, max_active_jobs=1), httpx.MockTransport(fail_one))
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=uploads)
        assert response.status_code == 200
        results = response.json()["results"]
        errors = response.json()["errors"]
        assert [r["text"] for r in results] == ["text 1", "text 3"]
        assert [r["index"] for r in results] == [0, 2]
        assert len(errors) == 1 and errors[0]["index"] == 1
        assert errors[0]["status_code"] == {"invalid_audio": 400, "backend_error": 502,
                                           "timeout": 504, "internal_error": 500}[failure]
        assert "error" in errors[0] and "text" not in errors[0]
        assert "private internal detail" not in response.text
        assert app.state.active_jobs == 0


def test_all_invalid_batch_has_per_file_errors():
    def unused_backend(request):
        pytest.fail("Invalid uploads must not reach the backend")

    app = create_app(Settings(vad_enabled=False), httpx.MockTransport(unused_backend))
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=[("file", ("empty.wav", b"")),
                                                                  ("file", ("wrong.wav", audio(rate=8000)))])
        assert response.status_code == 200
        assert response.json()["results"] == []
        assert [r["status_code"] for r in response.json()["errors"]] == [400, 400]
        assert app.state.active_jobs == 0


@pytest.mark.parametrize("backend_concurrency", [1, 2])
def test_batch_larger_than_job_capacity_runs_in_bounded_waves(backend_concurrency):
    async def run():
        active = peak = 0
        jobs_seen = []

        async def measured_backend(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            jobs_seen.append(app.state.active_jobs)
            try:
                await asyncio.sleep(0.01)
                return backend(request)
            finally:
                active -= 1

        app = create_app(Settings(vad_enabled=False, max_active_jobs=2,
                                  backend_concurrency=backend_concurrency), httpx.MockTransport(measured_backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                response = await client.post("/v1/audio/transcriptions", files=files(7))
                assert response.status_code == 200
                assert [r["text"] for r in response.json()["results"]] == [f"text {i}" for i in range(1, 8)]
                assert peak == backend_concurrency
                assert all(j == 2 for j in jobs_seen)
                assert app.state.active_jobs == 0

    asyncio.run(run())


def test_capacity_is_shared_with_other_requests_and_released_on_cancel():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        active = 0

        async def blocked_backend(request):
            nonlocal active
            active += 1
            entered.set()
            try:
                await release.wait()
                return backend(request)
            finally:
                active -= 1

        app = create_app(Settings(vad_enabled=False, max_active_jobs=1), httpx.MockTransport(blocked_backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                task = asyncio.create_task(client.post("/v1/audio/transcriptions", files=files(3)))
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    for count in (1, 2):
                        rejected = await client.post("/v1/audio/transcriptions", files=files(count))
                        assert rejected.status_code == 429
                        assert rejected.headers["retry-after"] == "5"
                        assert app.state.active_jobs == 1
                finally:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    release.set()
                assert active == 0
                assert app.state.active_jobs == 0
                accepted = await client.post("/v1/audio/transcriptions", files=files())
                assert accepted.status_code == 200
                assert app.state.active_jobs == 0

    asyncio.run(run())


@pytest.mark.parametrize("data,count,status", [
    ({"response_format": "text"}, 1, 400),
    ({"response_format": "text"}, 2, 400),
    ({"model": "unknown"}, 2, 400),
    ({"stream": "true"}, 2, 400),
    ({"audio_format": "mp3"}, 2, 422),
    ({}, 3, 400),
])
def test_request_validation(data, count, status):
    def unused_backend(request):
        pytest.fail("Rejected requests must not reach the backend")

    app = create_app(Settings(vad_enabled=False, max_batch_files=2), httpx.MockTransport(unused_backend))
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", data=data, files=files(count))
        assert response.status_code == status
        assert app.state.active_jobs == 0


def test_batch_upload_budget_applies_to_total_files():
    app = create_app(Settings(vad_enabled=False, max_upload_mb=1))
    with TestClient(app) as client:
        # Under the middleware's metadata allowance, but over the audio budget.
        response = client.post("/v1/audio/transcriptions", data={"audio_format": "pcm"},
                               files=[("file", ("a.pcm", b"\x00" * 600000)),
                                      ("file", ("b.pcm", b"\x00" * 600000))])
        assert response.status_code == 413
        assert app.state.active_jobs == 0


def test_batch_authentication_and_backend_key():
    def authenticated_backend(request):
        assert request.headers["authorization"] == "Bearer internal"
        return backend(request)

    app = create_app(Settings(vad_enabled=False, api_key="public", backend_api_key="internal"),
                     httpx.MockTransport(authenticated_backend))
    with TestClient(app) as client:
        assert client.post("/v1/audio/transcriptions", files=files(2)).status_code == 401
        response = client.post("/v1/audio/transcriptions", files=files(2),
                               headers={"Authorization": "Bearer public"})
        assert response.status_code == 200
        assert all(r["status_code"] == 200 for r in response.json()["results"])


def test_batch_limit_configuration_and_openapi(monkeypatch):
    monkeypatch.setenv("MAX_BATCH_FILES", "12")
    assert Settings.from_env().max_batch_files == 12
    with pytest.raises(ValueError, match="max_batch_files"):
        Settings(max_batch_files=0).validate()
    schema = create_app().openapi()
    body = schema["paths"]["/v1/audio/transcriptions"]["post"]["requestBody"]
    ref = body["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    properties = schema["components"]["schemas"][ref]["properties"]
    assert properties["file"]["type"] == "array"
    assert properties["audio_format"]["default"] == "wav"


def test_batch_silence_and_duplicate_names():
    def unused_backend(request):
        pytest.fail("Silence must not be submitted to vLLM")

    app = create_app(Settings(), httpx.MockTransport(unused_backend))
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions",
                               files=[("file", ("same.wav", audio(0))), ("file", ("same.wav", audio(0)))])
        assert response.status_code == 200
        assert response.json()["errors"] == []
        assert [r["index"] for r in response.json()["results"]] == [0, 1]
        assert all(r["text"] == "" and r["filename"] == "same.wav" for r in response.json()["results"])


def test_single_failure_uses_unified_errors_array():
    app = create_app(Settings(vad_enabled=False))
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files={"file": ("bad.wav", b"invalid")})
        assert response.status_code == 200
        assert response.json()["results"] == []
        assert len(response.json()["errors"]) == 1
        assert response.json()["errors"][0]["status_code"] == 400
        assert response.json()["errors"][0]["uttid"]
        assert app.state.active_jobs == 0


def test_single_and_batch_share_global_backend_limit():
    async def run():
        first_entered = asyncio.Event()
        release = asyncio.Event()
        active = peak = 0

        async def measured_backend(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                if not first_entered.is_set():
                    first_entered.set()
                    await release.wait()
                await asyncio.sleep(0.01)
                return backend(request)
            finally:
                active -= 1

        app = create_app(Settings(vad_enabled=False, max_active_jobs=3, backend_concurrency=1),
                         httpx.MockTransport(measured_backend))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                single = asyncio.create_task(client.post("/v1/audio/transcriptions", files=files()))
                batch = None
                try:
                    await asyncio.wait_for(first_entered.wait(), 2)
                    batch = asyncio.create_task(client.post("/v1/audio/transcriptions", files=files(5)))
                    async def admitted():
                        while app.state.active_jobs != 3:
                            await asyncio.sleep(0)
                    await asyncio.wait_for(admitted(), 2)
                    rejected = await client.post("/v1/audio/transcriptions", files=files())
                    assert rejected.status_code == 429
                    release.set()
                    responses = await asyncio.wait_for(asyncio.gather(single, batch), 2)
                    assert all(response.status_code == 200 for response in responses)
                    assert len(responses[1].json()["results"]) == 5
                    assert responses[1].json()["errors"] == []
                    assert peak == 1
                    assert app.state.active_jobs == 0
                finally:
                    release.set()
                    for task in (single, batch):
                        if task and not task.done():
                            task.cancel()
                    await asyncio.gather(*(t for t in (single, batch) if t), return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("count", [1, 2])
def test_uttid_is_preserved_for_success_and_file_errors(count):
    app = create_app(Settings(vad_enabled=False), httpx.MockTransport(backend))
    with TestClient(app) as client:
        for invalid in (False, True):
            uploads = files(count)
            if invalid:
                uploads[-1] = ("file", ("bad.wav", b"invalid"))
            ids = [f"audio_{i}" for i in range(count)]
            response = client.post("/v1/audio/transcriptions", files=uploads, data={"uttid": ids})
            assert response.status_code == 200
            body = response.json()
            assert set(body) == {"request_id", "results", "errors"}
            items = sorted(body["results"] + body["errors"], key=lambda item: item["index"])
            assert [item["uttid"] for item in items] == ids
            if invalid:
                assert body["errors"][0]["uttid"] == ids[-1]
            else:
                assert body["errors"] == []


@pytest.mark.parametrize("ids", [["only_one"], ["a", "a"], ["a", " "], ["", "b"]])
def test_invalid_uttids_reject_entire_request(ids):
    app = create_app(Settings(vad_enabled=False))
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=files(2), data={"uttid": ids})
        assert response.status_code == 400
        assert "detail" in response.json()
        assert "results" not in response.json()
        assert app.state.active_jobs == 0
