// The one and only PYBIND11_MODULE for dequant_cuda. Keeping it out of the
// .cu files means nvcc never has to compile pybind's templates.
#include "dequant_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one, "add one (CUDA)", py::arg("x"));

    m.def("gemv", &gemv, "packed-weight GEMV, fp16 [M,K] x [N,K]^T -> [M,N]",
          py::arg("x"), py::arg("qweight"), py::arg("scales"), py::arg("zeros"),
          py::arg("bits"), py::arg("group_size"), py::arg("symmetric"),
          py::arg("interleaved"));

    m.def("dequantize", &dequantize, "materialize a packed weight as fp16 [N,K]",
          py::arg("qweight"), py::arg("scales"), py::arg("zeros"), py::arg("bits"),
          py::arg("group_size"), py::arg("symmetric"), py::arg("interleaved"),
          py::arg("K"));
}
