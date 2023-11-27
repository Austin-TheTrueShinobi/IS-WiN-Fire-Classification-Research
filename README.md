# IS-WiN-Research
Research repository for the IS-WiN laboratory at Clemson University. Conducted research includes; inference, classification, and GPU-[CUDA] CNN accelerations.


## Run Details

### RUN1-Default

Improved Deeplabv3+ utilizing Mobilenetv3 network for decoding operations to increase segmentation speed. ASPP and DCNN modules used alongside ReLu activation functions.

### RUN2-CUDA-SuperClass-ReLu

Same as RUN1-Default except includes a curated CUDA optimizer for inference and quantization.

### RUN3-CUDA-ELU

Same as RUN2-CUDA-SuperClass-ReLu except utilizes ELU activation functions.

## CUDA Memory Usage Analysis

This section outlines the components of CUDA memory analysis and what to look for:

### Active Cache Timeline

- **Purpose:** The active cache timeline provides a timeline of events related to the GPU memory cache. This cache is responsible for caching recently freed GPU memory to expedite future allocations.
  
- **What to Look For:**
  - Frequent cache hits or misses: Frequent hits suggest efficient memory reuse, while frequent misses may indicate that the cache is not being utilized effectively.
  - Large cache allocations: If there are large allocations in the cache, it might impact the efficiency of the cache.

### Active Memory Timeline

- **Purpose:** The active memory timeline shows the timeline of allocations and deallocations of CUDA memory during the execution of your program.
  
- **What to Look For:**
  - Memory growth: Look for patterns of memory growth over time. Sudden spikes in memory usage might indicate a memory leak or inefficient memory management.
  - Frequent allocations and deallocations: Excessive allocations and deallocations may lead to memory fragmentation, which can impact overall performance.
  - Unusual patterns: Identify any unusual patterns or spikes in memory usage that might indicate unexpected behavior in your code.

### Allocator State History

- **Purpose:** The allocator state history records the history of allocation events, showing the sequence of operations that led to the current state of GPU memory.
  
- **What to Look For:**
  - Leaked allocations: Identify any allocations that are not followed by deallocations. These could be indicative of memory leaks.
  - Frequent reallocations: Excessive reallocations may indicate inefficient use of memory, potentially impacting performance.
  - Allocation patterns: Understand the allocation patterns to optimize memory usage. For example, reusing existing memory instead of frequently allocating new memory can be more efficient.

## C++ CUDA Optimizer for Inference and Quantization

Below is an example code snippet for a simple C++ CUDA optimizer for inference and quantization:

```cpp
// Include the necessary headers
#include <iostream>
#include <cuda_runtime.h>

// Define CUDA kernel for optimization
__global__ void optimizeKernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Perform optimization on input data
        input[idx] = input[idx] * 0.8; // Example: Quantization
    }
}

// CUDA optimizer function
void cudaOptimizer(float* input, int size) {
    // Allocate device memory
    float* d_input;
    cudaMalloc((void**)&d_input, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    optimizeKernel<<<gridSize, blockSize>>>(d_input, size);

    // Copy data back from device to host
    cudaMemcpy(input, d_input, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
}

int main() {
    // Example usage
    const int dataSize = 1024;
    float inputData[dataSize];

    // Initialize input data

    // Call CUDA optimizer
    cudaOptimizer(inputData, dataSize);

    // Process optimized data

    return 0;
}
