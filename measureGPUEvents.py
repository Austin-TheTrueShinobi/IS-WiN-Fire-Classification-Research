import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess

def measure_latency_throughput(model, dummy_input, repetitions=300, optimal_batch_size=100):
    device = torch.device("cuda")
    model.to(device)
    dummy_input = dummy_input.to(device)

    # GPU-WARM-UP
    for _ in range(5):
        _ = model(dummy_input)

    # MEASURE LATENCY
    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_latency = np.sum(timings) / repetitions
    std_latency = np.std(timings)

    # MEASURE THROUGHPUT
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time

    throughput = (repetitions * optimal_batch_size) / total_time

    return mean_latency, std_latency, throughput

# Load the saved PyTorch model
model_path = '/model.pt'
model = torch.jit.load(model_path)

# Set up dummy input
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float)

# Measure latency, throughput, and print results
mean_latency, std_latency, throughput = measure_latency_throughput(model, dummy_input)
print(f"Mean Latency: {mean_latency} ms, Std Deviation: {std_latency} ms")
print(f"Throughput: {throughput} images/second")

# Plotting detailed chart
batch_sizes = [2**i for i in range(1, 6)]
latencies = []
throughputs = []

# Arrays to store min and max values for mean latency
min_latencies = []
max_latencies = []

for batch_size in batch_sizes:
    dummy_input_batch = torch.randn(batch_size, 3, 224, 224, dtype=torch.float)
    mean_latency, std_latency, throughput = measure_latency_throughput(model, dummy_input_batch)
    latencies.append(mean_latency)
    throughputs.append(throughput)
    min_latencies.append(mean_latency - std_latency)
    max_latencies.append(mean_latency + std_latency)

# Plotting latency vs batch size
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(batch_sizes, latencies, marker='o', label='Mean Latency')
plt.fill_between(batch_sizes, min_latencies, max_latencies, color='gray', alpha=0.2, label='Min/Max Latency Range')
plt.xscale('log', base=2)
plt.xlabel('Batch Size')
plt.ylabel('Latency (ms)')
plt.legend()
plt.title('Mean Latency vs Batch Size')

# Plotting throughput vs batch size
plt.subplot(1, 2, 2)
plt.plot(batch_sizes, throughputs, marker='o', label='Throughput')
plt.xscale('log', base=2)
plt.xlabel('Batch Size')
plt.ylabel('Throughput (images/second)')
plt.legend()
plt.title('Throughput vs Batch Size')

# GPU Utilization (using nvidia-smi for illustration purposes)
# Run 'nvidia-smi dmon -s u -c 1' in the terminal to get GPU utilization values
gpu_utilization_command = "nvidia-smi dmon -s u -c 1"
gpu_utilization_output = subprocess.check_output(gpu_utilization_command, shell=True, text=True)

# Skip the header line and parse GPU utilization values
gpu_utilization_lines = gpu_utilization_output.strip().split('\n')[1:]
gpu_utilization_values = [
    float(line.split()[1]) for line in gpu_utilization_lines if line.strip() and line[0].isdigit()
]

plt.subplot(2, 2, 4)
plt.plot(batch_sizes[:len(gpu_utilization_values)], gpu_utilization_values, marker='o', label='GPU Utilization')
plt.xscale('log', base=2)
plt.xlabel('Batch Size')
plt.ylabel('GPU Utilization (%)')
plt.legend()
plt.title('GPU Utilization')

plt.tight_layout()
plt.show()

