#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <cstdint>
#include "model.hpp"
#include "shader.hpp"

class Renderer {
public:
    Renderer(int width = 1280, int height = 720);
    ~Renderer();

    /**
     * @brief Sube el frame de la cámara (BGR de OpenCV) a la VRAM.
     * Utiliza glTexSubImage2D para evitar reasignar memoria de textura en cada frame.
     */
    void update_camera_texture(const uint8_t* data, int width, int height);

    /**
     * @brief Pasada compuesta en GPU (2 pasadas en un solo FBO):
     * 1. Dibuja el fondo de la cámara en 2D sin depth test.
     * 2. Limpia el buffer de profundidad y dibuja el modelo 3D encima con depth test.
     */
    void render_scene(const Model& model, const Shader& modelShader, const Shader& bgShader);

    /**
     * @brief Extrae los píxeles compuestos finales en formato BGR.
     * Al leer directamente en GL_BGR, elimina la necesidad de cv2.cvtColor en Python.
     */
    std::vector<uint8_t> get_pixels();

    int get_width() const { return m_width; }
    int get_height() const { return m_height; }

private:
    GLFWwindow* window;
    int m_width, m_height;
    
    // Framebuffer y renderbuffers offscreen
    unsigned int FBO, textureColorBuffer, RBO;

    // Recursos para el fondo de cámara en GPU
    unsigned int quadVAO, quadVBO;
    unsigned int cameraTexture;

    void setupOffscreen();
    void setupCameraQuad();
};