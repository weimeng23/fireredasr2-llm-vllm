import io
import json
import sys
from email.parser import BytesParser
from email.policy import default

import pytest

from examples import client


@pytest.mark.parametrize("count", [1, 2])
def test_client_sends_repeated_file_fields(tmp_path, monkeypatch, capsys, count):
    paths = [tmp_path / f"audio {i}.pcm" for i in range(count)]
    for i, path in enumerate(paths):
        path.write_bytes(bytes([i, 0]) * 20)

    def respond(request, timeout):
        assert request.full_url == "http://test/v1/audio/transcriptions"
        assert timeout == 7200
        message = BytesParser(policy=default).parsebytes(
            f'Content-Type: {request.headers["Content-type"]}\r\n\r\n'.encode() + request.data
        )
        parts = list(message.iter_parts())
        uploads = [p for p in parts if p.get_param("name", header="content-disposition") == "file"]
        assert [p.get_filename() for p in uploads] == [p.name for p in paths]
        assert [p.get_payload(decode=True) for p in uploads] == [p.read_bytes() for p in paths]
        fields = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                  for p in parts if p.get_filename() is None}
        assert fields["audio_format"] == b"pcm"
        assert fields["response_format"] == b"json"
        body = {"results": [{"text": "ok"} for _ in range(count)], "errors": []}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(client.urllib.request, "urlopen", respond)
    monkeypatch.setattr(sys, "argv", ["client.py", *map(str, paths), "--url", "http://test", "--audio-format", "pcm"])
    client.main()
    assert "ok" in capsys.readouterr().out


@pytest.mark.parametrize("count", [1, 2])
def test_client_exits_nonzero_for_http_200_with_file_errors(tmp_path, monkeypatch, capsys, count):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"invalid")
    body = {"results": [{"index": 0, "text": "ok"}],
            "errors": [{"index": 1, "status_code": 400, "error": "Invalid WAV"}]}
    monkeypatch.setattr(client.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(sys, "argv", ["client.py", *([str(path)] * count)])
    with pytest.raises(SystemExit) as exc:
        client.main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == body
    assert "1 file(s) failed" in captured.err


def test_client_rejects_text_batch_before_reading_files(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["client.py", "a.wav", "b.wav", "--format", "text"])
    with pytest.raises(SystemExit) as exc:
        client.main()
    assert exc.value.code == 2
