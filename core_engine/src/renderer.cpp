#include "renderer.hpp"
#include <algorithm>
#include <stdexcept>

Renderer::Renderer(int width, int height) 
    : m_width(width), m_height(height), m_renderMode(0), quadVAO(0), quadVBO(0), cameraTexture(0) {
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

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        throw std::runtime_error("[ERROR] Fallo al inicializar GLAD");
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    
    setupOffscreen();
    setupCameraQuad();
}

void Renderer::setupOffscreen() {
    // Crear el Framebuffer
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);

    // Textura para el color final compuesto
    glGenTextures(1, &textureColorBuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0);

    // Buffer de profundidad para el z-buffer del modelo 3D
    glGenRenderbuffers(1, &RBO);
    glBindRenderbuffer(GL_RENDERBUFFER, RBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_width, m_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::setupCameraQuad() {
    float quadVertices[] = {
        // Posiciones     // UVs
        -1.0f,  1.0f,     0.0f, 0.0f, // Superior Izquierda
        -1.0f, -1.0f,     0.0f, 1.0f, // Inferior Izquierda
         1.0f, -1.0f,     1.0f, 1.0f, // Inferior Derecha

        -1.0f,  1.0f,     0.0f, 0.0f, // Superior Izquierda
         1.0f, -1.0f,     1.0f, 1.0f, // Inferior Derecha
         1.0f,  1.0f,     1.0f, 0.0f  // Superior Derecha
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);

    // Atributo 0: Posición vec2
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);

    // Atributo 1: Coordenadas de textura vec2
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<const void*>(2 * sizeof(float)));

    glBindVertexArray(0);

    // Textura donde se carga el stream de la cámara web
    glGenTextures(1, &cameraTexture);
    glBindTexture(GL_TEXTURE_2D, cameraTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Asignación inicial de memoria en VRAM (vacía)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

Renderer::~Renderer() {
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteTextures(1, &cameraTexture);
    glDeleteFramebuffers(1, &FBO);
    glDeleteTextures(1, &textureColorBuffer);
    glDeleteRenderbuffers(1, &RBO);
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Renderer::update_camera_texture(const uint8_t* data, int width, int height) {
    if (!data) return;

    glBindTexture(GL_TEXTURE_2D, cameraTexture);
    if (width == m_width && height == m_height) {
        // Subida directa de alta velocidad sin recrear la textura
        // Usar GL_BGR le indica a la GPU que OpenCV entrega bytes B-G-R,
        // permitiendo que el hardware reorganice a RGB en tiempo real a coste cero.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, data);
    } else {
        m_width = width;
        m_height = height;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::setRenderMode(int mode) {
    m_renderMode = std::clamp(mode, 0, 2);
}

void Renderer::render_scene(const Model& model, const Shader& modelShader, const Shader& bgShader) {
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    glViewport(0, 0, m_width, m_height);

    // PASADA 1: DIBUJAR FONDO DE CÁMARA (2D)
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    
    bgShader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, cameraTexture);
    bgShader.setInt("cameraTexture", 0);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    // PASADA 2: DIBUJAR CASCO 3D (AR)
    glClear(GL_DEPTH_BUFFER_BIT); // Limpiamos profundidad para el modelo
    glEnable(GL_DEPTH_TEST);      // Reactivamos depth test
    glEnable(GL_BLEND);           // Habilitar canal alfa para modo holograma
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    modelShader.use();
    modelShader.setInt("u_RenderMode", m_renderMode);
    modelShader.setFloat("u_Time", static_cast<float>(glfwGetTime()));
    model.draw(modelShader);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glfwPollEvents();
}

std::vector<uint8_t> Renderer::get_pixels() {
    std::vector<uint8_t> pixels(m_width * m_height * 3);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    // Leemos directamente en GL_BGR para que la salida ya coincida con OpenCV/PyQt
    glReadPixels(0, 0, m_width, m_height, GL_BGR, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return pixels;
}
