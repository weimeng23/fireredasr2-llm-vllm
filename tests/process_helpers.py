"""Importable worker functions for tests using the spawn start method."""
import os

from gateway.audio import prepare_audio


def controlled_prepare(state, lock, release, pids, *args):
    with lock:
        state['active'] += 1
        state['started'] += 1
        state['peak'] = max(state['peak'], state['active'])
        if os.getpid() not in pids:
            pids.append(os.getpid())
    try:
        if not release.wait(10):
            raise RuntimeError('Test did not release preprocessing worker')
        return prepare_audio(*args)
    finally:
        with lock:
            state['active'] -= 1
