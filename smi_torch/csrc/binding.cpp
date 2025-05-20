#include <torch/extension.h>
#include <vector>

// Declare CUDA functions
std::vector<at::Tensor> ksg_mi_cuda(
    at::Tensor x,
    at::Tensor y,
    int k_neighbors);

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ksg_mi", &ksg_mi_cuda, "KSG MI estimation (CUDA)",
          py::arg("x"), py::arg("y"), py::arg("k_neighbors"));
}