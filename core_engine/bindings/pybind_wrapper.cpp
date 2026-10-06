#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

// prueba
class Renderer {
public:
    Renderer() { 
        std::cout << "[C++] Renderer nativo inicializado en memoria.\n"; 
    }
    void ping() { 
        std::cout << "[C++] Pong! Enlace pybind11 funcionando a la perfeccion.\n"; 
    }
};

// Módulo de enlace. El nombre 'ar_helmet_core' debe coincidir con CMake.
PYBIND11_MODULE(ar_helmet_core, m) {
    m.doc() = "Core engine nativo para AR-Helmet";
    
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<>())
        .def("ping", &Renderer::ping);
}