#include "renderer.hpp"
#include <iostream>
#include <stdexcept>

Renderer::Renderer(int width, int height) : m_width(width), m_height(height){
    if (!glfwInit()) throw std::runtime_error("[ERROR] Fallo al inicializar GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    window = glfwCreateWindow(m_width, m_height, "AR Core", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("[ERROR] Fallo al crear la ventana GLFW oculta");
    }

    // Hacer que este hilo sea el dueño del contexto gráfico
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("[ERROR] Fallo al inicializar GLAD");
    }

    glEnable(GL_DEPTH_TEST);
    setupOffscreen();
}

void Renderer::setupOffscreen() {
    // 1. Crear el Framebuffer
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);

    // 2. Crear una textura vacía para guardar los colores renderizados
    glGenTextures(1, &textureColorBuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0);

    // 3. Crear buffer de profundidad (vital para el 3D)
    glGenRenderbuffers(1, &RBO);
    glBindRenderbuffer(GL_RENDERBUFFER, RBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_width, m_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO);

    // 4. Volver al framebuffer por defecto
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Renderer::~Renderer() {
    glDeleteFramebuffers(1, &FBO);
    glDeleteTextures(1, &textureColorBuffer);
    glDeleteRenderbuffers(1, &RBO);
    glfwDestroyWindow(window);
    glfwTerminate();
}

std::vector<uint8_t> Renderer::get_pixels() {
    std::vector<uint8_t> pixels(m_width * m_height * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    // Leer los bytes de la GPU a la CPU
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return pixels;
}

void Renderer::render(const Model& model) {
    // Redirigir el renderizado al FBO oculto
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    
    // Limpiar con fondo negro transparente
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    model.draw();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glfwPollEvents();
}

void Renderer::ping() {
    std::cout << "Pong!.\n";
}
