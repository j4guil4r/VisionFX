#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "model.hpp"

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void ping();

    /**
     * @brief Ejecuta una pasada de dibujado en la GPU.
     * Limpia los buffers de color y profundidad, y manda a rasterizar
     * los polígonos del modelo 3D proporcionado.
     * @param model Referencia constante al modelo 3D cargado en VRAM.
     */
    void render(const Model& model);

private:
    GLFWwindow* window;
};