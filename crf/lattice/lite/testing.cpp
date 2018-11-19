#include <torch/torch.h>
#include <ATen/ATen.h>
#include <vector>
#include <iostream>

at::Tensor d_sigmoid(at::Tensor z) {
  auto s = at::sigmoid(z);
  return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("d_sigmoid", &d_sigmoid, "testing__");
}