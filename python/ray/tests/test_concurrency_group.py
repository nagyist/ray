# coding: utf-8
import asyncio
import sys
import threading
import time
from typing import Any, Tuple

import pytest

import ray
from ray._common.utils import get_or_create_event_loop
from ray._private.test_utils import run_string_as_driver
from ray._common.test_utils import SignalActor


# This tests the methods are executed in the correct eventloop.
def test_basic(ray_start_regular_shared):
    @ray.remote(concurrency_groups={"io": 2, "compute": 4})
    class AsyncActor:
        def __init__(self):
            self.eventloop_f1 = None
            self.eventloop_f2 = None
            self.eventloop_f3 = None
            self.eventloop_f4 = None
            self.default_eventloop = get_or_create_event_loop()

        @ray.method(concurrency_group="io")
        async def f1(self):
            self.eventloop_f1 = get_or_create_event_loop()
            return threading.current_thread().ident

        @ray.method(concurrency_group="io")
        def f2(self):
            self.eventloop_f2 = get_or_create_event_loop()
            return threading.current_thread().ident

        @ray.method(concurrency_group="compute")
        def f3(self):
            self.eventloop_f3 = get_or_create_event_loop()
            return threading.current_thread().ident

        @ray.method(concurrency_group="compute")
        def f4(self):
            self.eventloop_f4 = get_or_create_event_loop()
            return threading.current_thread().ident

        def f5(self):
            # If this method is executed in default eventloop.
            assert get_or_create_event_loop() == self.default_eventloop
            return threading.current_thread().ident

        @ray.method(concurrency_group="io")
        def do_assert(self):
            if self.eventloop_f1 != self.eventloop_f2:
                return False
            if self.eventloop_f3 != self.eventloop_f4:
                return False
            if self.eventloop_f1 == self.eventloop_f3:
                return False
            if self.eventloop_f1 == self.eventloop_f4:
                return False
            return True

    ###############################################
    a = AsyncActor.remote()
    f1_thread_id = ray.get(a.f1.remote())  # executed in the "io" group.
    f2_thread_id = ray.get(a.f2.remote())  # executed in the "io" group.
    f3_thread_id = ray.get(a.f3.remote())  # executed in the "compute" group.
    f4_thread_id = ray.get(a.f4.remote())  # executed in the "compute" group.

    assert f1_thread_id == f2_thread_id
    assert f3_thread_id == f4_thread_id
    assert f1_thread_id != f3_thread_id

    assert ray.get(a.do_assert.remote())

    assert ray.get(a.f5.remote())  # executed in the default group.

    # It also has the ability to specify it at runtime.
    # This task will be invoked in the `compute` thread pool.
    result = ray.get(a.f2.options(concurrency_group="compute").remote())
    assert result == f3_thread_id


# The case tests that the asyncio count down works well in one concurrency
# group.
def test_async_methods_in_concurrency_group(ray_start_regular_shared):
    @ray.remote(concurrency_groups={"async": 3})
    class AsyncBatcher:
        def __init__(self):
            self.batch = []
            self.event = None

        @ray.method(concurrency_group="async")
        def init_event(self):
            self.event = asyncio.Event()
            return True

        @ray.method(concurrency_group="async")
        async def add(self, x):
            self.batch.append(x)
            if len(self.batch) >= 3:
                self.event.set()
            else:
                await self.event.wait()
            return sorted(self.batch)

    a = AsyncBatcher.remote()
    ray.get(a.init_event.remote())

    x1 = a.add.remote(1)
    x2 = a.add.remote(2)
    x3 = a.add.remote(3)
    r1 = ray.get(x1)
    r2 = ray.get(x2)
    r3 = ray.get(x3)
    assert r1 == [1, 2, 3]
    assert r1 == r2 == r3


# This case tests that if blocking task in default group blocks
# tasks in other groups.
# See https://github.com/ray-project/ray/issues/20475
def test_default_concurrency_group_does_not_block_others(ray_start_regular_shared):
    @ray.remote(concurrency_groups={"my_group": 1})
    class AsyncActor:
        def __init__(self):
            pass

        async def f1(self):
            time.sleep(10000)
            return "never return"

        @ray.method(concurrency_group="my_group")
        def f2(self):
            return "ok"

    async_actor = AsyncActor.remote()
    async_actor.f1.remote()
    assert "ok" == ray.get(async_actor.f2.remote())


# This case tests that a blocking group doesn't blocks
# tasks in other groups.
# See https://github.com/ray-project/ray/issues/19593
def test_blocking_group_does_not_block_others(ray_start_regular_shared):
    @ray.remote(concurrency_groups={"group1": 1, "group2": 1})
    class AsyncActor:
        def __init__(self):
            pass

        @ray.method(concurrency_group="group1")
        async def f1(self):
            time.sleep(10000)
            return "never return"

        @ray.method(concurrency_group="group2")
        def f2(self):
            return "ok"

    async_actor = AsyncActor.remote()
    # Execute f1 twice for blocking the group1.
    obj_0 = async_actor.f1.remote()
    obj_1 = async_actor.f1.remote()
    # Wait a while to make sure f2 is scheduled after f1.
    ray.wait([obj_0, obj_1], timeout=5)
    # f2 should work well even if group1 is blocking.
    assert "ok" == ray.get(async_actor.f2.remote())


def test_system_concurrency_group(ray_start_regular_shared):
    @ray.remote
    class NormalActor:
        def block_forever(self):
            time.sleep(9999)
            return "never"

        def ping(self):
            return "pong"

    n = NormalActor.remote()
    n.block_forever.options(concurrency_group="_ray_system").remote()
    print(ray.get(n.ping.remote()))


@ray.remote(concurrency_groups={"io": 1, "compute": 1})
class Actor:
    def __init__(self):
        self._thread_local_data = threading.local()

    def set_thread_local(self, value: Any) -> int:
        self._thread_local_data.value = value
        return threading.current_thread().ident

    def get_thread_local(self) -> Tuple[Any, int]:
        return self._thread_local_data.value, threading.current_thread().ident


class TestThreadingLocalData:
    """
    This test verifies that synchronous tasks can access thread-local data that
    was set by previous synchronous tasks when the concurrency group has only
    one thread. For concurrency groups with multiple threads, it doesn't promise
    access to the same thread-local data because Ray currently doesn't expose APIs
    for users to specify which thread the task will be scheduled on in the same
    concurrency group.
    """

    def test_tasks_on_default_executor(self, ray_start_regular_shared):
        a = Actor.remote()
        tid_1 = ray.get(a.set_thread_local.remote("f1"))
        value, tid_2 = ray.get(a.get_thread_local.remote())
        assert tid_1 == tid_2
        assert value == "f1"

    def test_tasks_on_specific_executor(self, ray_start_regular_shared):
        a = Actor.remote()
        tid_1 = ray.get(a.set_thread_local.options(concurrency_group="io").remote("f1"))
        value, tid_2 = ray.get(
            a.get_thread_local.options(concurrency_group="io").remote()
        )
        assert tid_1 == tid_2
        assert value == "f1"

    def test_tasks_on_different_executors(self, ray_start_regular_shared):
        a = Actor.remote()
        tid_1 = ray.get(a.set_thread_local.options(concurrency_group="io").remote("f1"))
        tid_3 = ray.get(
            a.set_thread_local.options(concurrency_group="compute").remote("f2")
        )
        value, tid_2 = ray.get(
            a.get_thread_local.options(concurrency_group="io").remote()
        )
        assert tid_1 == tid_2
        assert value == "f1"

        value, tid_4 = ray.get(
            a.get_thread_local.options(concurrency_group="compute").remote()
        )
        assert tid_3 == tid_4
        assert value == "f2"


def test_multiple_threads_in_same_group(ray_start_regular_shared):
    """
    This test verifies that all threads in the same concurrency group are still
    alive from the Python interpreter's perspective even if Ray tasks have finished, so that
    thread-local data will not be garbage collected.
    """

    @ray.remote
    class Actor:
        def __init__(self, signal: SignalActor, max_concurrency: int):
            self._thread_local_data = threading.local()
            self.signal = signal
            self.thread_id_to_data = {}
            self.max_concurrency = max_concurrency

        def set_thread_local(self, value: int) -> int:
            # If the thread-local data were garbage collected after the previous
            # task on the same thread finished, `self.data` would be incremented
            # more than once for the same thread.
            assert not hasattr(self._thread_local_data, "value")
            self._thread_local_data.value = value
            self.thread_id_to_data[threading.current_thread().ident] = value
            ray.get(self.signal.wait.remote())

        def check_thread_local_data(self) -> bool:
            assert len(self.thread_id_to_data) == self.max_concurrency
            assert hasattr(self._thread_local_data, "value")
            assert (
                self._thread_local_data.value
                == self.thread_id_to_data[threading.current_thread().ident]
            )
            ray.get(self.signal.wait.remote())

    max_concurrency = 5
    signal = SignalActor.remote()
    a = Actor.options(max_concurrency=max_concurrency).remote(signal, max_concurrency)

    refs = []
    for i in range(max_concurrency):
        refs.append(a.set_thread_local.remote(i))

    ray.get(signal.send.remote())
    ray.get(refs)

    refs = []
    for _ in range(max_concurrency):
        refs.append(a.check_thread_local_data.remote())

    ray.get(signal.send.remote())
    ray.get(refs)


def test_invalid_concurrency_group():
    """Verify that when a concurrency group has max concurrency set to 0,
    an error is raised when the actor is created. This test uses
    `run_string_as_driver` and checks whether the error message appears in the
    driver's stdout. Since the error in the core worker process does not raise
    an exception in the driver process, we need to check the driver process's
    stdout.
    """

    script = """
import ray

ray.init()

@ray.remote(concurrency_groups={"io": 0, "compute": 0})
class A:
    def __init__(self):
        pass

actor = A.remote()
    """

    output = run_string_as_driver(script)
    assert "max_concurrency must be greater than 0" in output


if __name__ == "__main__":

    sys.exit(pytest.main(["-sv", __file__]))
