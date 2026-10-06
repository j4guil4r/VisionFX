#include <pybind11/pybind11.h>
#include <iostream>
#include "shader.hpp"
#include "renderer.hpp"

namespace py = pybind11;


PYBIND11_MODULE(ar_helmet_core, m) {
    m.doc() = "Core engine nativo para AR-Helmet";
    
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<>())
        .def("ping", &Renderer::ping);
}