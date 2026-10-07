#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cstdint>
#include "model.hpp"

class Renderer {
public:
    Renderer(int width = 1280, int height = 720);
    ~Renderer();
    
    void ping();

    /**
     * @brief Ejecuta una pasada de dibujado en la GPU.
     * Limpia los buffers de color y profundidad, y manda a rasterizar
     * los polígonos del modelo 3D proporcionado.
     * @param model Referencia constante al modelo 3D cargado en VRAM.
     */
    void render(const Model& model);

    /**
     * @brief Extrae los píxeles renderizados de la memoria de video.
     * @return Vector de bytes (RGB) del fotograma actual.
     */
    std::vector<uint8_t> get_pixels();

private:
    GLFWwindow* window;
    int m_width, m_height;
    
    // Identificadores del Framebuffer oculto
    unsigned int FBO, textureColorBuffer, RBO;
    
    void setupOffscreen();
};