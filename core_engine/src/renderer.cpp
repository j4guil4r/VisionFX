#include "renderer.hpp"
#include <iostream>
#include <stdexcept>

Renderer::Renderer() {
    if (!glfwInit()) throw std::runtime_error("[ERROR] Fallo al inicializar GLFW");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Nota: Mantenemos la ventana de GLFW oculta. 
    // Python se encargará de mostrar los resultados.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    // Creacion del contexto (ventana invisible de 1280x720)
    window = glfwCreateWindow(1280, 720, "AR Helmet Core", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("[ERROR] Fallo al crear la ventana GLFW oculta");
    }

    // Hacer que este hilo sea el dueño del contexto gráfico
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("[ERROR] Fallo al inicializar GLAD");
    }
    
    std::cout << "[C++] Contexto OpenGL 3.3 Core e inicializador GLAD listos.\n";
}

Renderer::~Renderer() {
    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "[C++] Contexto GLFW destruido limpiamente.\n";
}

void Renderer::ping() {
    std::cout << "[C++] Pong! El Renderer real esta escuchando a Python.\n";
}

void Renderer::render(const Model& model) {
    glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // TODO: llamadas a los Shaders (Fase 3)
    // shader.use();
    // shader.setMat4("view", matrix...);

    // Despachar los buffers de vértices a la placa de video
    model.draw();

    // Intercambiar la memoria oculta con la visible (Double Buffering)
    glfwSwapBuffers(window);
    
    // Procesar eventos del sistema operativo para evitar que la ventana se congele
    glfwPollEvents();
}