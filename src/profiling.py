import time 
from collections import deque


class InferenceProfiler:

    def __init__(self, window_size: int = 100) -> None:
        if window_size < 1:
            raise ValueError("Window size mist be positive")

        self.latencies_ms = deque(maxlen=window_size)

    def start(self) -> float:
        """
        Return a high-resolution start timestamp.

        """
        return time.perf_counter()

    def stop(self, start_time: float) -> float:
        """
        Record one duration and return its latency in milliseconds.

        """

        latency_ms = (time.perf_counter()-start_time)*1000
        self.latencies_ms.append(latency_ms)

        return latency_ms

    @property
    def average_latency_ms(self) -> float:

        if not self.latencies_ms:
            return 0.0 

        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def fps(self) -> float:

        latency = self.average_latency_ms

        if latency <= 1.0:
            return 0.0 

        return 1000.0 / latency

        