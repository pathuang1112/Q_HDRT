import os
import threading
import time
import tracemalloc


def start_memory_tracing():
    """Start Python allocation tracing for peak-memory reporting."""
    if not tracemalloc.is_tracing():
        tracemalloc.start()


def get_memory_usage():
    current_mb = None
    peak_mb = None
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

    rss_mb = None
    try:
        import psutil

        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        pass

    return {
        "traced_current_mb": current_mb,
        "traced_peak_mb": peak_mb,
        "rss_mb": rss_mb,
    }


class PeakMemoryMonitor:
    """Continuously sample process RSS while retaining tracemalloc metrics."""

    def __init__(self, sample_interval=0.05):
        self.sample_interval = sample_interval
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._process = None
        self._baseline_rss_mb = None
        self._peak_rss_mb = None

    def start(self):
        start_memory_tracing()
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._baseline_rss_mb = self._read_rss_mb()
            self._peak_rss_mb = self._baseline_rss_mb
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except Exception:
            self._process = None
        return self

    def _read_rss_mb(self):
        if self._process is None:
            return None
        return self._process.memory_info().rss / 1024 / 1024

    def _sample_once(self):
        try:
            rss_mb = self._read_rss_mb()
        except Exception:
            return None
        if rss_mb is not None:
            with self._lock:
                if self._peak_rss_mb is None or rss_mb > self._peak_rss_mb:
                    self._peak_rss_mb = rss_mb
        return rss_mb

    # Sample process RSS independently of Python allocation tracing.
    def _sample_loop(self):
        while not self._stop_event.wait(self.sample_interval):
            self._sample_once()

    def snapshot(self):
        usage = get_memory_usage()
        rss_mb = self._sample_once()
        with self._lock:
            peak_rss_mb = self._peak_rss_mb
        usage["rss_mb"] = rss_mb if rss_mb is not None else usage["rss_mb"]
        usage["peak_rss_mb"] = peak_rss_mb
        if peak_rss_mb is not None and self._baseline_rss_mb is not None:
            usage["peak_rss_delta_mb"] = max(0.0, peak_rss_mb - self._baseline_rss_mb)
        return usage

    def stop(self):
        self._sample_once()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval * 4))


def elapsed_since(start_time):
    return time.perf_counter() - start_time


def format_time_costs(iter_num, total_time, preprocessing_time, online_selection_time,
                      preprocessing_memory=None, total_memory=None):
    parts = [
        f"iternum={iter_num}",
        f"Total time={total_time}",
        f"total_time={total_time}",
        f"preprocessing_time={preprocessing_time}",
        f"online_selection_time={online_selection_time}",
    ]

    if preprocessing_memory:
        for key, value in preprocessing_memory.items():
            if value is not None:
                parts.append(f"preprocessing_{key}={value}")
    if total_memory:
        for key, value in total_memory.items():
            if value is not None:
                parts.append(f"total_{key}={value}")

    return "\n".join(parts) + "\n"
