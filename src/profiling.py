import time 
import numpy as np 
from collections import defaultdict


class InferencePerformance:
    pass 


class PerformanceMonitor:

    def __init__(self, report_interval=5) -> None:
        self.report_interval = report_interval
        self.latencies = defaultdict(list)
        self.frame_count = 0
        self.start_time = time.perf_counter()
    
    def record(self, name, elapsed_time):
        self.latencies[name].append(elapsed_time)    

    def next_frame(self):
        self.frame_count += 1
        elapsed = time.perf_counter() - self.start_time

        if elapsed >= self.report_interval:
            self.report(elapsed)

    def report(self, elapsed):
        print(f"\n================ Performance Report =======================")

        for name, value in self.latencies.items():
            print(
                f"{name}: "
                f"mean: {np.mean(value):.2f} ms | "
                f"p50: {np.percentile(value, 50):.2f} ms | "
                f"p90: {np.percentile(value, 90):.3f} ms | "
                f"p99: {np.percentile(value, 99):.3f} ms"
            )


        print(f"Pipeline FPS: {self.frame_count / elapsed:.2f}")

        self.latencies.clear()
        self.frame_count = 0
        self.start_time = time.perf_counter()







        