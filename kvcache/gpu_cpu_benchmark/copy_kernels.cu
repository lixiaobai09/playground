#include <cstdint>
#include <sys/cdefs.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>


void copy_dma(torch::Tensor& src, torch::Tensor& dst,
              const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream);
  }
}

template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    T* ptr;
    auto st = cudaHostGetDevicePointer(
        (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
    TORCH_CHECK(st == cudaSuccess,
                "Host tensor not registered/pinned (or bad ptr)");
    return ptr;
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}

template<typename DType>
__global__ void copy_kernel(
    DType* __restrict__ src_ptr,
    DType* __restrict__ dst_ptr,
    const int64_t* __restrict__ block_mapping,
    const int words_per_block){
  const int block_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int64_t src_block_number = block_mapping[2 * block_id];
  const int64_t dst_block_number = block_mapping[2 * block_id + 1];
  DType* src = src_ptr + src_block_number * words_per_block;
  DType* dst = dst_ptr + dst_block_number * words_per_block;

  for (int i = tid; i < words_per_block; i += num_threads) {
    dst[i] = src[i];
  }
}

template<typename DType>
void issue_kernel_copy(DType* __restrict__ src_ptr,
                       DType* __restrict__ dst_prt,
                       const int64_t* __restrict__ block_mapping,
                       const int64_t block_size_bytes,
                       const int num_blocks,
                       const cudaStream_t stream
                      ) {
  const int words_per_block = block_size_bytes / sizeof(DType);
  const dim3 grid(num_blocks);
  const dim3 block(std::min(words_per_block, 128));
  copy_kernel<<<grid, block, 0, stream>>>(
    src_ptr, dst_prt, block_mapping, words_per_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_custom_kernel(torch::Tensor& src, torch::Tensor& dst,
                        const torch::Tensor& block_mapping,
                        const bool use_vec = false) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
	  
  int num_blocks = block_mapping.size(0);
  const int64_t block_size_bytes = src.element_size() * src.stride(0);

  const int64_t* block_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(block_mapping.cuda());
  void* src_ptr = get_kernel_ptr<void, torch::Tensor>(src);
  void* dst_ptr = get_kernel_ptr<void, torch::Tensor>(dst);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (use_vec) {
    issue_kernel_copy<int4>(
      reinterpret_cast<int4*>(src_ptr),
      reinterpret_cast<int4*>(dst_ptr),
      block_mapping_ptr,
      block_size_bytes,
      num_blocks,
      stream);
  } else {
    issue_kernel_copy<ulonglong2>(
      reinterpret_cast<ulonglong2*>(src_ptr),
      reinterpret_cast<ulonglong2*>(dst_ptr),
      block_mapping_ptr,
      block_size_bytes,
      num_blocks,
      stream);
  }

}
