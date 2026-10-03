#include <torch/extension.h>
#include <string>
#include <vector>

// Declare CUDA functions
std::vector<at::Tensor> ksg_statistics_cuda(
    at::Tensor x,
    at::Tensor y,
    int64_t k_neighbors,
    const std::string& algorithm);

std::vector<at::Tensor> ksg_mi_cuda(
    at::Tensor x,
    at::Tensor y,
    int64_t k_neighbors,
    const std::string& algorithm);

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ksg_statistics", &ksg_statistics_cuda,
          "KSG building blocks (CUDA): returns [eps, counts_x, counts_y] where eps is the "
          "k-th nearest neighbour distance in the joint space and counts are the numbers "
          "of other samples strictly within eps in each marginal space. Inputs have shape "
          "(N, D) or (B, N, D).",
          py::arg("x"), py::arg("y"), py::arg("k_neighbors"), py::arg("algorithm") = "auto");
    m.def("ksg_mi", &ksg_mi_cuda,
          "KSG MI estimation (CUDA): returns [mi, mi_std] as float64 tensors. Inputs have "
          "shape (N, D) or (B, N, D).",
          py::arg("x"), py::arg("y"), py::arg("k_neighbors"), py::arg("algorithm") = "auto");
}
