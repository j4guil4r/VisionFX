#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <glm/gtc/type_ptr.hpp>
#include <stdexcept>

#include "shader.hpp"
#include "renderer.hpp"
#include "model.hpp"
#include "LightEstimator.hpp"

namespace py = pybind11;


PYBIND11_MODULE(ar_helmet_core, m) {
    m.doc() = "Core engine nativo para VisionFX con estimacion de iluminacion";
    
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<int, int>(), py::arg("width") = 1280, py::arg("height") = 720)
        .def("ping", &Renderer::ping)
        .def("render", &Renderer::render)
        // Convertir std::vector a numpy.ndarray de (H, W, 3)
        .def("get_frame", [](Renderer& self) {
            auto pixels = self.get_pixels();
            // Creamos un array numpy con las dimensiones del FBO
            py::array_t<uint8_t> result({720, 1280, 3});
            auto req = result.request();
            std::memcpy(req.ptr, pixels.data(), pixels.size());
            return result;
        });
    
    py::class_<Model>(m, "Model")
        .def(py::init<const std::string &>());
    
    py::class_<Shader>(m, "Shader")
        .def(py::init<const char*, const char*>())
        .def("use", &Shader::use)
        // Puente que recibe una lista plana de 16 floats desde Python
        .def("set_mat4", [](const Shader& self, const std::string& name, const std::vector<float>& value) {
            if (value.size() != 16) throw std::runtime_error("[C++] La matriz debe tener exactamente 16 elementos.");
            // make_mat4 lee los 16 floats y construye la matriz de OpenGL
            glm::mat4 mat = glm::make_mat4(value.data());
            self.setMat4(name, mat);
        })
        .def("set_vec3", &Shader::setVec3)
        .def("set_float", &Shader::setFloat);

    py::class_<LightEstimator>(m, "LightEstimator")
        .def(py::init<>())
        .def("estimate", [](LightEstimator& self, py::array_t<uint8_t> frame_np) {
            py::buffer_info buf = frame_np.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::runtime_error("El frame debe ser un array 3D con 3 canales (H, W, 3).");
            }
            // Mapeo directo a cv::Mat sin duplicar memoria
            cv::Mat mat(static_cast<int>(buf.shape[0]), static_cast<int>(buf.shape[1]), CV_8UC3, buf.ptr);
            LightEstimate est = self.estimate(mat);

            // Retorna una tupla: (direction_list, intensity, ambient_list)
            return py::make_tuple(
                std::vector<float>{est.direction[0], est.direction[1], est.direction[2]},
                est.intensity,
                std::vector<float>{est.ambientColor[0], est.ambientColor[1], est.ambientColor[2]}
            );
        });
}