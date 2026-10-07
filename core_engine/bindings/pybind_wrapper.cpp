#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>

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
        .def("use", &Shader::use)
        // NUEVO: Puente que recibe una lista plana de 16 floats desde Python
        .def("set_mat4", [](const Shader& self, const std::string& name, const std::vector<float>& value) {
            if (value.size() != 16) throw std::runtime_error("[C++] La matriz debe tener exactamente 16 elementos.");
            // make_mat4 lee los 16 floats y construye la matriz de OpenGL
            glm::mat4 mat = glm::make_mat4(value.data());
            self.setMat4(name, mat);
        });
}