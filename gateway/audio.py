"""Validate 16 kHz mono PCM16/WAV, then detect speech and slice in memory."""
import io
import struct
import time
import wave

import webrtcvad

RATE = 16000
SAMPLE_BYTES = 2


class AudioError(ValueError):
    pass


def decode(data: bytes, max_seconds: float, audio_format: str) -> bytes:
    """Extract PCM without resampling; raw PCM parameters are a caller contract."""
    if audio_format not in ("pcm", "wav"):
        raise AudioError("audio_format must be pcm or wav")
    if not data:
        raise AudioError("Audio is empty")
    if audio_format == "wav":
        try:
            with wave.open(io.BytesIO(data), "rb") as src:
                if (src.getframerate(), src.getnchannels(), src.getsampwidth(),
                        src.getcomptype()) != (RATE, 1, SAMPLE_BYTES, "NONE"):
                    raise AudioError("WAV must be 16000 Hz, mono, uncompressed PCM16")
                frames = src.getnframes()
                if frames > max_seconds * RATE:
                    raise AudioError(f"Audio exceeds {max_seconds:g} seconds")
                # Read one extra frame to also catch an incomplete final sample.
                pcm = src.readframes(frames + 1)
                if len(pcm) != frames * SAMPLE_BYTES:
                    raise AudioError("Truncated or invalid WAV sample data")
        except (wave.Error, EOFError, struct.error, RuntimeError) as exc:
            raise AudioError("Invalid WAV; expected 16000 Hz mono PCM16 WAV") from exc
    else:
        # Raw PCM has no metadata: rate, channels and byte order cannot be inferred.
        pcm = data
    if not pcm:
        raise AudioError("Audio is empty")
    if len(pcm) % SAMPLE_BYTES:
        raise AudioError("PCM16 data must contain complete 2-byte samples")
    if len(pcm) > max_seconds * RATE * SAMPLE_BYTES:
        raise AudioError(f"Audio exceeds {max_seconds:g} seconds")
    return pcm


def detect_speech(pcm: bytes, mode=1, frame_ms=20, silence_ms=500, padding_ms=200,
                  timeout=180):
    """Return padded speech intervals in seconds over in-memory PCM16.

    A fresh VAD owns each recording. Keep short utterances; silence_ms is a
    closing hangover, not a minimum speech length. A partial final frame is
    zero-padded for classification only.
    """
    if mode not in range(4) or frame_ms not in (10, 20, 30):
        raise ValueError("Invalid WebRTC VAD mode or frame size")
    if silence_ms <= 0 or padding_ms < 0 or timeout <= 0:
        raise ValueError("Invalid VAD duration")
    if len(pcm) % SAMPLE_BYTES:
        raise AudioError("Truncated normalized audio")
    vad = webrtcvad.Vad(mode)
    deadline = time.monotonic() + timeout
    raw_intervals = []
    total = len(pcm) // SAMPLE_BYTES
    frame_samples = RATE * frame_ms // 1000
    hangover = round(RATE * silence_ms / 1000)
    padding = round(RATE * padding_ms / 1000)
    start = None
    last_voice_end = 0
    offset = 0
    while offset < total:
        if time.monotonic() >= deadline:
            raise AudioError("VAD preprocessing timed out")
        end = min(offset + frame_samples, total)
        frame = pcm[offset * SAMPLE_BYTES:end * SAMPLE_BYTES]
        if vad.is_speech(frame.ljust(frame_samples * SAMPLE_BYTES, b"\0"), RATE):
            if start is None:
                start = offset
            last_voice_end = end
        elif start is not None and end - last_voice_end >= hangover:
            raw_intervals.append((start, last_voice_end))
            start = None
        offset = end
    if start is not None:
        raw_intervals.append((start, last_voice_end))

    merged = []
    for start, end in raw_intervals:
        start, end = max(0, start - padding), min(total, end + padding)
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [(start / RATE, end / RATE) for start, end in merged]


def plan_chunks(duration, speech, max_seconds=25.0):
    """Cover the original timeline once; only speech regions go to ASR."""
    if not 0 < max_seconds <= 30:
        raise ValueError("Require 0 < max_seconds <= 30")
    rate = 16000
    total, limit = round(duration * rate), max(1, round(max_seconds * rate))
    chunks = []

    def append_region(start, end, silent):
        while start < end:
            stop = min(start + limit, end)
            chunks.append((start / rate, stop / rate, silent))
            start = stop

    cursor = 0
    for left, right in speech:
        start, end = round(left * rate), round(right * rate)
        if not cursor <= start < end <= total:
            raise ValueError("Speech intervals must be ordered, disjoint and in bounds")
        append_region(cursor, start, True)
        append_region(start, end, False)
        cursor = end
    append_region(cursor, total, True)
    return chunks


def cut_wav(pcm: bytes, start: float, end: float) -> bytes:
    """Wrap one slice of in-memory PCM as a WAV payload for the backend."""
    first = round(start * RATE)
    last = min(len(pcm) // SAMPLE_BYTES, round(end * RATE))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(SAMPLE_BYTES)
        dst.setframerate(RATE)
        dst.writeframes(pcm[first * SAMPLE_BYTES:last * SAMPLE_BYTES])
    return buffer.getvalue()


def prepare_audio(data: bytes, settings, audio_format: str):
    """Decode once, then hand back the plan; slices are cut on demand."""
    pcm = decode(data, settings.max_audio_seconds, audio_format)
    duration = len(pcm) / SAMPLE_BYTES / RATE
    speech = detect_speech(
        pcm, settings.vad_mode, settings.vad_frame_ms, settings.vad_silence_ms,
        settings.vad_padding_ms, settings.media_timeout,
    ) if settings.vad_enabled else [(0, duration)]
    return duration, pcm, plan_chunks(duration, speech, settings.chunk_seconds)


def join_text(parts):
    result = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if result and result[-1].isascii() and part[0].isascii():
            result += " "
        result += part
    return result
