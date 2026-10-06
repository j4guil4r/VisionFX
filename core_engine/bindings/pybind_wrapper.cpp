#include <pybind11/pybind11.h>
#include <iostream>
#include "shader.hpp"

namespace py = pybind11;

// prueba
class Renderer {
public:
    Renderer() { 
        std::cout << "[C++] Renderer nativo inicializado en memoria.\n";
        Shader testShader("dummy.vert", "dummy.frag");
    }
    void ping() { 
        std::cout << "[C++] Pong! Enlace pybind11 funcionando a la perfeccion.\n"; 
    }
};


PYBIND11_MODULE(ar_helmet_core, m) {
    m.doc() = "Core engine nativo para AR-Helmet";
    
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<>())
        .def("ping", &Renderer::ping);
}