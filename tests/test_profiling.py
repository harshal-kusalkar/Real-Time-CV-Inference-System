import time 

from src.profiling import InferenceProfiler

profiler = InferenceProfiler(window_size=10)

for _ in range(10):

    start = profiler.start()

    time.sleep(0.005)

    latency = profiler.stop(start_time=start)
    print(f"Latency : {latency:.2f} ms")


print(f"Avg Latency: {profiler.average_latency_ms} ms")
print(f"FPS: {profiler.fps:.2f} operations/sec")

assert profiler.average_latency_ms > 0
assert profiler.fps > 0

print("Profiler test passed!")

