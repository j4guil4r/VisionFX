#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

class Shader {
public:
    // El ID numérico que OpenGL le asigna a este programa de shaders en la VRAM
    unsigned int ID;

    /**
     * @brief Constructor del Shader.
     * Lee los archivos de texto del disco, los compila en la GPU y los enlaza
     * en un único programa ejecutable. Si hay errores de sintaxis en GLSL, los
     * imprime en la consola.
     * @param vertexPath Ruta al archivo del Vertex Shader (.vert)
     * @param fragmentPath Ruta al archivo del Fragment Shader (.frag)
     */
    Shader(const char* vertexPath, const char* fragmentPath);
    ~Shader();

    /**
     * @brief Activa este shader en la GPU.
     * Cualquier llamada de dibujado que ocurra después de esto usará este programa.
     */
    void use() const;

    /**
     * @brief Inyecta una matriz matemática de 4x4 en el shader.
     * Vital para enviar las matrices de Proyección, Vista (cámara) y Modelo
     * generadas por MediaPipe/OpenCV hacia la tarjeta de video.
     * @param name Nombre de la variable `uniform` dentro del código GLSL.
     * @param mat La matriz 4x4 de GLM a enviar.
     */
    void setMat4(const std::string &name, const glm::mat4 &mat) const;

    // Funciones útiles para inyectar variables (uniforms) a la GPU
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;

private:
    void checkCompileErrors(unsigned int shader, std::string type) const;
};