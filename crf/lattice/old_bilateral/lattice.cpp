#include <torch/torch.h>
#include "permutohedral.h"
#include "Image.h"

at::Tensor testfunc(at::Tensor z) {
    return z;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("testfunc", &testfunc, "lattice testfunc");
}