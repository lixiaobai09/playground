#include <pybind11/pybind11.h>
#include "cache.h"
#include <torch/torch.h>
#include <iostream>

PYBIND11_MODULE(gpu_cpu_benchmark, m) {
  m.def("copy_dma", &copy_dma);
  m.def("copy_custom_kernel", &copy_custom_kernel, pybind11::arg("src"),
        pybind11::arg("dst"), pybind11::arg("block_mapping"),
        pybind11::arg("use_vec") = false);
}
