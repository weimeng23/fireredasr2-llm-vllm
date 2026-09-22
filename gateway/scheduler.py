"""Round-robin scheduling of ready audio files, with bounded fragment work."""
import asyncio
from collections import deque
from dataclasses import dataclass, field


async def run_in_thread(function, *args):
    return await finish_on_cancel(asyncio.create_task(asyncio.to_thread(function, *args)))


async def run_in_process(pool, function, *args):
    return await finish_on_cancel(asyncio.get_running_loop().run_in_executor(pool, function, *args))


async def finish_on_cancel(task):
    """Keep the slot until underlying thread/process work really finishes."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Cancelling a Future does not stop its running executor job.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Retrieve a possible worker error after cancellation.
        raise


@dataclass(eq=False)
class AudioWork:
    pcm: bytes
    pending: deque
    texts: list
    future: asyncio.Future
    tasks: set = field(default_factory=set)


class FragmentScheduler:
    def __init__(self, concurrency, recognize):
        self.concurrency = concurrency
        self.recognize = recognize
        self.ready = deque()
        self.running = set()
        self.wakeup = asyncio.Event()

    async def __aenter__(self):
        self.dispatcher = asyncio.create_task(self._dispatch())
        return self

    async def __aexit__(self, *exc):
        self.dispatcher.cancel()
        await asyncio.gather(self.dispatcher, return_exceptions=True)
        # The application drains file submissions before closing this scheduler.
        assert not self.running and not self.ready

    async def submit(self, pcm, plan):
        pending = deque((i, start, end) for i, (start, end, silent) in enumerate(plan) if not silent)
        texts = [""] * len(plan)
        if not pending:
            return texts
        work = AudioWork(pcm, pending, texts, asyncio.get_running_loop().create_future())
        self.ready.append(work)
        self.wakeup.set()
        try:
            return await work.future
        finally:
            # On failure or cancellation, remove pending fragments and drain siblings.
            if work in self.ready:
                self.ready.remove(work)
            tasks = list(work.tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch(self):
        while True:
            await self.wakeup.wait()
            self.wakeup.clear()
            while self.ready and len(self.running) < self.concurrency:
                work = self.ready.popleft()
                if work.future.done():
                    continue
                index, start, end = work.pending.popleft()
                if work.pending:
                    self.ready.append(work)
                task = asyncio.create_task(self._recognize(work, index, start, end))
                self.running.add(task)
                work.tasks.add(task)
                # A callback also releases slots for tasks cancelled before starting.
                task.add_done_callback(lambda done, work=work: self._finished(work, done))

    async def _recognize(self, work, index, start, end):
        try:
            work.texts[index] = await self.recognize(work.pcm, start, end)
        except Exception as exc:
            if not work.future.done():
                work.future.set_exception(exc)

    def _finished(self, work, task):
        self.running.discard(task)
        work.tasks.discard(task)
        if not work.pending and not work.tasks and not work.future.done():
            work.future.set_result(work.texts)
        self.wakeup.set()
