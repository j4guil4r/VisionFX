#pragma once
#include <string>

// Clase para abstraer la compilación de código GLSL en la GPU
class Shader {
public:
    unsigned int ID;
    Shader(const char* vertexPath, const char* fragmentPath);
    ~Shader();

    // Activar el shader para usarlo en el ciclo de renderizado
    void use() const;

    // Funciones útiles para inyectar variables (uniforms) a la GPU
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;

private:
    void checkCompileErrors(unsigned int shader, std::string type) const;
};