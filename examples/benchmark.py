#!/usr/bin/env python3
"""Benchmark the gateway's single/batch transcription API using the standard library."""
import argparse
import io
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_audio(path, audio_format):
    data = path.read_bytes()
    if audio_format == 'pcm':
        samples = len(data) // 2
        if not data or len(data) % 2:
            raise ValueError(f'{path}: expected nonempty 16 kHz mono PCM16LE')
    else:
        with wave.open(io.BytesIO(data), 'rb') as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, 'NONE'):
                raise ValueError(f'{path}: WAV must be 16 kHz mono uncompressed PCM16')
            samples = source.getnframes()
            if samples <= 0 or len(source.readframes(samples + 1)) != samples * 2:
                raise ValueError(f'{path}: empty or truncated WAV')
    return {'path': str(path), 'data': data, 'duration_s': samples / 16000}


def load_records(manifest, paths, audio_format):
    entries = []
    if manifest:
        for number, line in enumerate(manifest.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f'{manifest}:{number}: expected a JSON object')
            # Accept the existing LID (uttid/wav) and PPL (uid/wav_path) manifests.
            name = item.get('wav', item.get('wav_path'))
            uid = item.get('uttid', item.get('uid', str(number)))
            if not isinstance(name, str) or not name.strip() or not isinstance(uid, str) or not uid.strip():
                raise ValueError(f'{manifest}:{number}: expected string wav/wav_path and uttid/uid')
            path = Path(name).expanduser()
            entries.append((uid, path if path.is_absolute() else manifest.parent / path))
    else:
        entries = [(str(i), path.expanduser()) for i, path in enumerate(paths)]
    if not entries:
        raise ValueError('No audio files to benchmark')
    records, cache = [], {}
    for uid, path in entries:
        path = path.resolve()
        if path not in cache:
            cache[path] = load_audio(path, audio_format)
        records.append({'uttid': uid, **cache[path]})
    return records


def multipart(records, request_index, batch_size, audio_format, run_id):
    boundary = uuid.uuid4().hex
    chunks, expected = [], {}

    def field(name, value):
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.extend([value.encode(), b'\r\n'])

    field('audio_format', audio_format)
    field('response_format', 'json')
    for index in range(batch_size):
        record = records[(request_index * batch_size + index) % len(records)]
        uttid = f'{run_id}-{request_index}-{index}'
        expected[uttid] = {'index': index, 'duration_s': record['duration_s'], 'source_uttid': record['uttid']}
        field('uttid', uttid)
        filename = Path(record['path']).name.replace('\r', '%0D').replace('\n', '%0A').replace('"', '%22').replace('\\', '%5C')
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                      'Content-Type: application/octet-stream\r\n\r\n'.encode())
        chunks.extend([record['data'], b'\r\n'])
    chunks.append(f'--{boundary}--\r\n'.encode())
    return b''.join(chunks), f'multipart/form-data; boundary={boundary}', expected


def validate_response(body, expected):
    if not isinstance(body, dict) or not isinstance(body.get('results'), list) or not isinstance(body.get('errors'), list):
        raise ValueError('Expected gateway results/errors arrays; native vLLM responses are not supported')
    seen, successful, errors = set(), [], []
    for kind in ('results', 'errors'):
        for item in body[kind]:
            if not isinstance(item, dict):
                raise ValueError('Response item must be an object')
            uid = item.get('uttid')
            if not isinstance(uid, str) or uid not in expected or uid in seen:
                raise ValueError('Response contains an unknown or duplicate uttid')
            seen.add(uid)
            if type(item.get('index')) is not int or item['index'] != expected[uid]['index']:
                raise ValueError('Response index does not match uttid')
            status = item.get('status_code')
            if kind == 'results':
                if type(status) is not int or status != 200 or not isinstance(item.get('text'), str):
                    raise ValueError('Invalid successful result')
                successful.append(uid)
            else:
                if type(status) is not int or not 400 <= status <= 599 or 'error' not in item:
                    raise ValueError('Invalid per-file error')
                errors.append({'uttid': uid, 'source_uttid': expected[uid]['source_uttid'],
                               'status_code': status, 'error': str(item['error'])[:500]})
    if seen != set(expected):
        raise ValueError('Response is missing audio results')
    return successful, errors


def send_request(opener, url, records, index, batch_size, audio_format, run_id, timeout, api_key):
    started = time.perf_counter()
    outcome = {'request_index': index, 'http_status': None, 'items_succeeded': 0,
               'audio_seconds_succeeded': 0.0, 'item_error_counts': {}, 'errors': []}
    try:
        payload, content_type, expected = multipart(records, index, batch_size, audio_format, run_id)
        headers = {'Content-Type': content_type}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        request = urllib.request.Request(url, data=payload, headers=headers, method='POST')
        with opener.open(request, timeout=timeout) as response:
            outcome['http_status'] = response.status
            body = json.load(response)
        successful, errors = validate_response(body, expected)
        outcome['items_succeeded'] = len(successful)
        outcome['audio_seconds_succeeded'] = sum(expected[uid]['duration_s'] for uid in successful)
        outcome['item_error_counts'] = dict(Counter(str(error['status_code']) for error in errors))
        outcome['errors'] = errors[:10]
    except urllib.error.HTTPError as exc:
        outcome['http_status'] = exc.code
        outcome['errors'] = [{'error_type': 'HTTPError', 'error': exc.read(2048).decode('utf-8', errors='replace')}]
    except Exception as exc:
        outcome['errors'] = [{'error_type': type(exc).__name__, 'error': str(exc)[:500]}]
    outcome['latency_s'] = time.perf_counter() - started
    outcome['success'] = outcome['items_succeeded'] == batch_size and not outcome['errors']
    return outcome


def execute(records, *, url, concurrency, requests, batch_size, audio_format, timeout, api_key, trust_env):
    if requests == 0:
        return [], 0.0
    workers = min(concurrency, requests)
    gate = threading.Barrier(workers + 1)
    run_id = uuid.uuid4().hex

    def worker(worker_index):
        # One opener per thread; match the PPL benchmark's proxy bypass by default.
        opener = urllib.request.build_opener() if trust_env else urllib.request.build_opener(urllib.request.ProxyHandler({}))
        gate.wait()
        return [send_request(opener, url, records, index, batch_size, audio_format, run_id, timeout, api_key)
                for index in range(worker_index, requests, workers)]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, index) for index in range(workers)]
        started = time.perf_counter()
        gate.wait()
        outcomes = [outcome for future in futures for outcome in future.result()]
        elapsed = time.perf_counter() - started
    return outcomes, elapsed


def latency_summary(outcomes):
    values = sorted(outcome['latency_s'] * 1000 for outcome in outcomes)
    if not values:
        return None
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return round(values[low] + (values[high] - values[low]) * (position - low), 3)
    return {'min': round(values[0], 3), 'mean': round(statistics.fmean(values), 3),
            'p50': percentile(0.5), 'p95': percentile(0.95), 'p99': percentile(0.99),
            'max': round(values[-1], 3)}


def summarize(outcomes, elapsed, batch_size):
    good = [outcome for outcome in outcomes if outcome['success']]
    audio_s = sum(outcome['audio_seconds_succeeded'] for outcome in outcomes)
    items = sum(outcome['items_succeeded'] for outcome in outcomes)
    total_items = len(outcomes) * batch_size
    error_counts = Counter()
    errors = []
    for outcome in sorted(outcomes, key=lambda item: item['request_index']):
        error_counts.update(outcome['item_error_counts'])
        for error in outcome['errors']:
            if len(errors) < 10:
                errors.append({'request_index': outcome['request_index'], **error})
    return {'elapsed_s': round(elapsed, 6), 'requests_total': len(outcomes),
            'requests_succeeded': len(good), 'requests_failed': len(outcomes) - len(good),
            'requests_partial': sum(0 < item['items_succeeded'] < batch_size for item in outcomes),
            'request_error_rate': round(1 - len(good) / len(outcomes), 6),
            'requests_per_s': round(len(good) / elapsed, 3),
            'items_total': total_items, 'items_succeeded': items, 'items_failed': total_items - items,
            'item_error_rate': round(1 - items / total_items, 6), 'items_per_s': round(items / elapsed, 3),
            'audio_seconds_succeeded': round(audio_s, 6), 'audio_seconds_per_s': round(audio_s / elapsed, 3),
            'aggregate_rtf': round(elapsed / audio_s, 6) if audio_s else None,
            'request_latency_ms': latency_summary(outcomes),
            'successful_request_latency_ms': latency_summary(good),
            'http_status_counts': dict(Counter(str(item['http_status']) if item['http_status'] is not None else 'no_response'
                                               for item in outcomes)),
            'item_error_status_counts': dict(error_counts), 'errors': errors}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path, help='JSONL: uttid/wav (LID) or uid/wav_path (PPL); relative paths use manifest directory')
    source.add_argument('--audio', type=Path, nargs='+', help='Audio files to cycle through')
    parser.add_argument('--url', default='http://127.0.0.1:12345/v1/audio/transcriptions', help='Full gateway endpoint, including proxy prefix if any')
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4, 8, 16, 32], help='Concurrent HTTP requests per run')
    parser.add_argument('--request-batch-size', type=int, nargs='+', default=[1], help='Audio files per HTTP request; multiple values create a sweep')
    count = parser.add_mutually_exclusive_group()
    count.add_argument('--requests', type=int, help='HTTP requests per run (default: 100 when --repeats is omitted)')
    count.add_argument('--repeats', type=int, help='Dataset passes per run; request count is rounded up to full batches')
    parser.add_argument('--warmup-requests', '--warmup', type=int, default=3, help='Warmup HTTP requests before each run; excluded from statistics')
    parser.add_argument('--timeout', type=float, default=300, help='Socket timeout in seconds')
    parser.add_argument('--audio-format', choices=('wav', 'pcm'), default='wav')
    parser.add_argument('--trust-env', action='store_true', help='Use proxy settings from the environment')
    parser.add_argument('--output', type=Path, help='Write the full JSON report; authorization comes from ASR_API_KEY')
    args = parser.parse_args(argv)
    if min(args.concurrency + args.request_batch_size) < 1 or args.warmup_requests < 0:
        parser.error('concurrency/batch size must be positive; warmup must be nonnegative')
    if any(value is not None and value < 1 for value in (args.requests, args.repeats)):
        parser.error('requests/repeats must be positive')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')
    parsed = urllib.parse.urlsplit(args.url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username is not None or parsed.password is not None:
        parser.error('url must be an HTTP(S) endpoint without embedded credentials; use ASR_API_KEY')
    return args


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.manifest, args.audio, args.audio_format)
    report = {'schema_version': 1, 'url': args.url, 'audio_format': args.audio_format,
              'dataset': {'manifest': str(args.manifest) if args.manifest else None,
                          'records': len(records), 'unique_files': len({row['path'] for row in records}),
                          'audio_seconds': sum(row['duration_s'] for row in records)},
              'reports': []}
    for batch_size in args.request_batch_size:
        requests = args.requests or (math.ceil(len(records) * args.repeats / batch_size) if args.repeats else 100)
        for concurrency in args.concurrency:
            options = dict(url=args.url, concurrency=concurrency, batch_size=batch_size,
                           audio_format=args.audio_format, timeout=args.timeout,
                           api_key=os.getenv('ASR_API_KEY', ''), trust_env=args.trust_env)
            print(f'Running concurrency={concurrency}, batch={batch_size}, requests={requests}', file=sys.stderr, flush=True)
            if requests < concurrency:
                print('Fewer requests than concurrency: some worker slots will remain unused.', file=sys.stderr)
            warmup, _ = execute(records, requests=args.warmup_requests, **options)
            if any(not item['success'] for item in warmup):
                raise RuntimeError(f'Warmup failed: {next(item["errors"] for item in warmup if not item["success"])}')
            outcomes, elapsed = execute(records, requests=requests, **options)
            result = {'arguments': {'concurrency': concurrency, 'requests': requests,
                                    'request_batch_size': batch_size, 'warmup_requests': args.warmup_requests,
                                    'timeout_s': args.timeout, 'trust_env': args.trust_env},
                      'summary': summarize(outcomes, elapsed, batch_size)}
            report['reports'].append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return int(any(run['summary']['requests_failed'] for run in report['reports']))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, wave.Error, EOFError) as exc:
        print(f'Benchmark failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
