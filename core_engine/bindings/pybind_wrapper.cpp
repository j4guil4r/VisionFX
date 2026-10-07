#include <pybind11/pybind11.h>
#include <iostream>
#include "shader.hpp"
#include "renderer.hpp"
#include "model.hpp"

namespace py = pybind11;


PYBIND11_MODULE(ar_helmet_core, m) {
    m.doc() = "Core engine nativo para AR-Helmet";
    
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<>())
        .def("ping", &Renderer::ping)
        .def("render", &Renderer::render);
    
    py::class_<Model>(m, "Model")
        .def(py::init<const std::string &>());
    
    py::class_<Shader>(m, "Shader")
        .def(py::init<const char*, const char*>())
        .def("use", &Shader::use);
}