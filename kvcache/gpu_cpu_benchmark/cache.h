#pragma once

#include <torch/all.h>

#include <map>
#include <vector>

void copy_dma(torch::Tensor& src, torch::Tensor& dst,
              const torch::Tensor& block_mapping);

void copy_custom_kernel(torch::Tensor& src, torch::Tensor& dst,
                        const torch::Tensor& block_mapping,
                        bool use_vec = false);
