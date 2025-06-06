import logging
import os
import platform
import sys
import time

import ray  # noqa F401

# Import ray before psutil will make sure we use psutil's bundled version
from ray._common.utils import get_system_memory

import psutil  # noqa E402

logger = logging.getLogger(__name__)


def get_rss(memory_info):
    """Get the estimated non-shared memory usage from psutil memory_info."""
    mem = memory_info.rss
    # OSX doesn't have the shared attribute
    if hasattr(memory_info, "shared"):
        mem -= memory_info.shared
    return mem


def get_shared(virtual_memory):
    """Get the estimated shared memory usage from psutil virtual mem info."""
    # OSX doesn't have the shared attribute
    if hasattr(virtual_memory, "shared"):
        return virtual_memory.shared
    else:
        return 0


def get_top_n_memory_usage(n: int = 10):
    """Get the top n memory usage of the process

    Params:
        n: Number of top n process memory usage to return.
    Returns:
        (str) The formatted string of top n process memory usage.
    """
    proc_stats = []
    for proc in psutil.process_iter(["memory_info", "cmdline"]):
        try:
            proc_stats.append(
                (get_rss(proc.info["memory_info"]), proc.pid, proc.info["cmdline"])
            )
        except psutil.NoSuchProcess:
            # We should skip the process that has exited. Refer this
            # issue for more detail:
            # https://github.com/ray-project/ray/issues/14929
            continue
        except psutil.AccessDenied:
            # On MacOS, the proc_pidinfo call (used to get per-process
            # memory info) fails with a permission denied error when used
            # on a process that isn’t owned by the same user. For now, we
            # drop the memory info of any such process, assuming that
            # processes owned by other users (e.g. root) aren't Ray
            # processes and will be of less interest when an OOM happens
            # on a Ray node.
            # See issue for more detail:
            # https://github.com/ray-project/ray/issues/11845#issuecomment-849904019  # noqa: E501
            continue
    proc_str = "PID\tMEM\tCOMMAND"
    for rss, pid, cmdline in sorted(proc_stats, reverse=True)[:n]:
        proc_str += "\n{}\t{}GiB\t{}".format(
            pid, round(rss / (1024**3), 2), " ".join(cmdline)[:100].strip()
        )
    return proc_str


class RayOutOfMemoryError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

    @staticmethod
    def get_message(used_gb, total_gb, threshold):
        proc_str = get_top_n_memory_usage(n=10)
        return (
            "More than {}% of the memory on ".format(int(100 * threshold))
            + "node {} is used ({} / {} GB). ".format(
                platform.node(), round(used_gb, 2), round(total_gb, 2)
            )
            + f"The top 10 memory consumers are:\n\n{proc_str}"
            + "\n\nIn addition, up to {} GiB of shared memory is ".format(
                round(get_shared(psutil.virtual_memory()) / (1024**3), 2)
            )
            + "currently being used by the Ray object store.\n---\n"
            "--- Tip: Use the `ray memory` command to list active "
            "objects in the cluster.\n"
            "--- To disable OOM exceptions, set "
            "RAY_DISABLE_MEMORY_MONITOR=1.\n---\n"
        )


class MemoryMonitor:
    """Helper class for raising errors on low memory.

    This presents a much cleaner error message to users than what would happen
    if we actually ran out of memory.

    The monitor tries to use the cgroup memory limit and usage if it is set
    and available so that it is more reasonable inside containers. Otherwise,
    it uses `psutil` to check the memory usage.

    The environment variable `RAY_MEMORY_MONITOR_ERROR_THRESHOLD` can be used
    to overwrite the default error_threshold setting.

    Used by test only. For production code use memory_monitor.cc
    """

    def __init__(self, error_threshold=0.95, check_interval=1):
        # Note: it takes ~50us to check the memory usage through psutil, so
        # throttle this check at most once a second or so.
        self.check_interval = check_interval
        self.last_checked = 0
        try:
            self.error_threshold = float(
                os.getenv("RAY_MEMORY_MONITOR_ERROR_THRESHOLD")
            )
        except (ValueError, TypeError):
            self.error_threshold = error_threshold
        # Try to read the cgroup memory limit if it is available.
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "rb") as f:
                self.cgroup_memory_limit_gb = int(f.read()) / (1024**3)
        except IOError:
            self.cgroup_memory_limit_gb = sys.maxsize / (1024**3)
        if not psutil:
            logger.warning(
                "WARNING: Not monitoring node memory since `psutil` "
                "is not installed. Install this with "
                "`pip install psutil` to enable "
                "debugging of memory-related crashes."
            )
        self.disabled = (
            "RAY_DEBUG_DISABLE_MEMORY_MONITOR" in os.environ
            or "RAY_DISABLE_MEMORY_MONITOR" in os.environ
        )

    def get_memory_usage(self):
        from ray._private.utils import get_used_memory

        total_gb = get_system_memory() / (1024**3)
        used_gb = get_used_memory() / (1024**3)

        return used_gb, total_gb

    def raise_if_low_memory(self):
        if self.disabled:
            return

        if time.time() - self.last_checked > self.check_interval:
            self.last_checked = time.time()
            used_gb, total_gb = self.get_memory_usage()

            if used_gb > total_gb * self.error_threshold:
                raise RayOutOfMemoryError(
                    RayOutOfMemoryError.get_message(
                        used_gb, total_gb, self.error_threshold
                    )
                )
            else:
                logger.debug(f"Memory usage is {used_gb} / {total_gb}")
