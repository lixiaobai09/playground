import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from gpu_cpu_benchmark import copy_dma, copy_custom_kernel


BYTES_TO_ALLOCATE = 4 << 30
NUM_BLOCKS_TO_COPY = 1000

# block sizes test (single direction)
BLOCK_SIZES_IN_KB_TEST = tuple(1 << i for i in range(2, 15))

# multiple concurrent directions test
BLOCK_SIZE = 2 << 20

gpu_tensor = torch.zeros(BYTES_TO_ALLOCATE, dtype=torch.int8, device="cuda:0")
cpu_tensor = torch.zeros(
    BYTES_TO_ALLOCATE,
    dtype=torch.int8,
    device="cpu",
    pin_memory=True
)

# test single direction transfer with various block sizes
results = {
    "GPU -> CPU": {"DMA": [], "Custom Kernel": []},
    "CPU -> GPU": {"DMA": [], "Custom Kernel": []}
}

for block_size_kb in BLOCK_SIZES_IN_KB_TEST:
    print("Testing block size", block_size_kb, "KB\n")

    block_size = block_size_kb << 10
    num_blocks = BYTES_TO_ALLOCATE // block_size
    bytes_copied = block_size * NUM_BLOCKS_TO_COPY

    gpu_tensor_view = gpu_tensor.view(-1, block_size)
    cpu_tensor_view = cpu_tensor.view(-1, block_size)

    transfer_directions = (
        (gpu_tensor_view, cpu_tensor_view, torch.cuda.Stream(), "GPU -> CPU"),
        (cpu_tensor_view, gpu_tensor_view, torch.cuda.Stream(), "CPU -> GPU"),
    )

    blocks_to_copy = np.linspace(
        0, num_blocks - 1, NUM_BLOCKS_TO_COPY, dtype=np.int64
    )
    src_to_dst = np.stack((blocks_to_copy, blocks_to_copy), axis=1)  # (i, i)
    src_to_dst_tensor = torch.from_numpy(src_to_dst)

    for src_tensor, dst_tensor, stream, direction_name in transfer_directions:
        print(src_tensor.device, "->", dst_tensor.device)
        for copy_function in (copy_dma, copy_custom_kernel):
            start_time = time.perf_counter()
            with torch.cuda.stream(stream):
                copy_function(src_tensor, dst_tensor, src_to_dst_tensor)
            stream.synchronize()
            total_time = time.perf_counter() - start_time
            throughput_gb = (bytes_copied / total_time) / (1 << 30)
            print(f"{copy_function.__name__:20}{throughput_gb:8.3f} GB/s")

            func_name = "DMA" if "dma" in copy_function.__name__ else "Custom Kernel"
            results[direction_name][func_name].append(throughput_gb)
        print()
    print()

# Plotting
for direction, data in results.items():
    plt.figure()
    plt.plot(BLOCK_SIZES_IN_KB_TEST, data["DMA"], label="DMA")
    plt.plot(BLOCK_SIZES_IN_KB_TEST, data["Custom Kernel"], label="Custom Kernel")
    plt.xlabel("Block Size (KB)")
    plt.xscale("log", base=2)
    plt.ylabel("Throughput (GB/s)")
    plt.title(f"Throughput with Block Size ({direction})")
    plt.legend()
    plt.grid(True)
    filename = f"{direction.replace(' -> ', '_to_').lower()}.svg"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")

# test concurrent transfers
'''

gpu_tensor_view = gpu_tensor.view(-1, BLOCK_SIZE)
cpu_tensor_view = cpu_tensor.view(-1, BLOCK_SIZE)
num_blocks = BYTES_TO_ALLOCATE // BLOCK_SIZE
bytes_copied = BLOCK_SIZE * NUM_BLOCKS_TO_COPY

copy_functions_tuples = (
    (copy_dma, copy_dma),
    (copy_custom_kernel, copy_custom_kernel),
    (copy_dma, copy_custom_kernel),
    (copy_custom_kernel, copy_dma)
)

for copy_functions in copy_functions_tuples:
    function_names = ", ".join(x.__name__ for x in copy_functions)
    print(f"Testing concurrent {copy_functions[0].__name__} GPU->CPU"
          f" and {copy_functions[1].__name__} CPU->GPU")

    for percent in range(1, 100):
        gpu_to_cpu_blocks_count = NUM_BLOCKS_TO_COPY * percent // 100
        cpu_to_gpu_block_count = NUM_BLOCKS_TO_COPY - gpu_to_cpu_blocks_count

        gpu_blocks_to_copy = np.linspace(
            0, num_blocks // 2 - 1, gpu_to_cpu_blocks_count, dtype=np.int64
        )
        gpu_to_cpu = np.stack((gpu_blocks_to_copy, gpu_blocks_to_copy), axis=1)
        gpu_to_cpu_tensor = torch.from_numpy(gpu_to_cpu)

        cpu_blocks_to_copy = np.linspace(
            num_blocks // 2,
            num_blocks - 1,
            cpu_to_gpu_block_count,
            dtype=np.int64
        )
        cpu_to_gpu = np.stack((cpu_blocks_to_copy, cpu_blocks_to_copy), axis=1)
        cpu_to_gpu_tensor = torch.from_numpy(cpu_to_gpu)

        gpu_to_cpu_stream = torch.cuda.Stream()
        cpu_to_gpu_stream = torch.cuda.Stream()

        transfer_directions = (
            (gpu_tensor_view, cpu_tensor_view, gpu_to_cpu_tensor, gpu_to_cpu_stream),
            (cpu_tensor_view, gpu_tensor_view, cpu_to_gpu_tensor, cpu_to_gpu_stream),
        )

        # submit transfers
        start_time = time.perf_counter()
        for i, direction in enumerate(transfer_directions):
            src_tensor, dst_tensor, src_to_dst, stream = direction
            with torch.cuda.stream(stream):
                copy_functions[i](src_tensor, dst_tensor, src_to_dst)

        # wait for both transfers to finish
        gpu_to_cpu_stream.synchronize()
        cpu_to_gpu_stream.synchronize()
        total_time = time.perf_counter() - start_time
        throughput_gb = (bytes_copied / total_time) / (1 << 30)
        print(f"GPU->CPU {percent}%: {throughput_gb:8.3f} GB/s")
    print()

'''