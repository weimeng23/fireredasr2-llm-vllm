import asyncio
import threading

import pytest

from gateway.scheduler import FragmentScheduler, run_in_thread


async def eventually(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(poll(), 2)


def test_single_audio_fills_backend_slots_and_reassembles_out_of_order():
    async def run():
        entered = []
        gates = [asyncio.Event() for _ in range(5)]
        finished = []

        async def recognize(pcm, start, end):
            index = int(start)
            entered.append(index)
            await gates[index].wait()
            finished.append(index)
            return str(index)

        async with FragmentScheduler(3, recognize) as scheduler:
            result = asyncio.create_task(scheduler.submit(b'a', [(i, i + 1, False) for i in range(5)]))
            await eventually(lambda: len(entered) == 3)
            assert entered == [0, 1, 2]
            gates[2].set()
            await eventually(lambda: len(entered) == 4)
            gates[3].set()
            await eventually(lambda: len(entered) == 5)
            for gate in gates:
                gate.set()
            assert await result == ['0', '1', '2', '3', '4']
            assert finished[:2] == [2, 3]
            assert not scheduler.running and not scheduler.ready
    asyncio.run(run())


def test_ready_audio_files_take_turns_including_late_arrivals():
    async def run():
        started = []
        gates = {}

        async def recognize(pcm, start, end):
            key = (pcm, int(start))
            started.append(key)
            gate = gates.setdefault(key, asyncio.Event())
            await gate.wait()
            return str(start)

        async with FragmentScheduler(1, recognize) as scheduler:
            a = asyncio.create_task(scheduler.submit(b'A', [(i, i + 1, False) for i in range(4)]))
            await eventually(lambda: len(started) == 1)
            b = asyncio.create_task(scheduler.submit(b'B', [(0, 1, False), (1, 2, False)]))
            c = asyncio.create_task(scheduler.submit(b'C', [(0, 1, False)]))
            await eventually(lambda: len(scheduler.ready) == 3)
            expected = [(b'A', 0), (b'A', 1), (b'B', 0), (b'C', 0),
                        (b'A', 2), (b'B', 1), (b'A', 3)]
            for i, key in enumerate(expected):
                await eventually(lambda: len(started) == i + 1)
                assert started[i] == key
                gates[key].set()
            await asyncio.gather(a, b, c)
    asyncio.run(run())


@pytest.mark.parametrize('cancel', [False, True])
def test_failure_or_cancel_drains_siblings_and_skips_pending_fragments(cancel):
    async def run():
        started = []
        active = set()
        fail = asyncio.Event()

        async def recognize(pcm, start, end):
            if pcm == b'B':
                return 'ok'
            started.append(start)
            active.add(start)
            try:
                if start == 0:
                    await fail.wait()
                    raise ValueError('bad fragment')
                await asyncio.Event().wait()
            finally:
                active.remove(start)

        async with FragmentScheduler(2, recognize) as scheduler:
            a = asyncio.create_task(scheduler.submit(b'A', [(i, i + 1, False) for i in range(10)]))
            await eventually(lambda: len(started) == 2)
            if cancel:
                a.cancel()
            else:
                fail.set()
            with pytest.raises(asyncio.CancelledError if cancel else ValueError):
                await a
            assert not active and started == [0, 1]
            assert not scheduler.ready and not scheduler.running
            assert await scheduler.submit(b'B', [(0, 1, False)]) == ['ok']
    asyncio.run(run())


def test_silence_is_not_submitted_and_keeps_original_positions():
    async def run():
        async def recognize(pcm, start, end):
            assert start == 1 and end == 2
            return 'speech'
        async with FragmentScheduler(2, recognize) as scheduler:
            assert await scheduler.submit(b'a', [(0, 1, True), (1, 2, False), (2, 3, True)]) == ['', 'speech', '']
            assert await scheduler.submit(b'a', [(0, 1, True)]) == ['']
    asyncio.run(run())


def test_thread_cancellation_waits_for_real_cpu_work_even_on_repeated_cancel():
    async def run():
        entered = threading.Event()
        release = threading.Event()
        def work():
            entered.set()
            assert release.wait(3)
            raise ValueError('thread failure after cancellation')
        task = asyncio.create_task(run_in_thread(work))
        try:
            await eventually(entered.is_set)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
