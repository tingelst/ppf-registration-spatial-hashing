#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(example, m) { m.def("add", &add); }
