#include "shader.hpp"
#include <iostream>

Shader::Shader(const char* vertexPath, const char* fragmentPath) {
    // TODO: lectura de archivos y las llamadas a glCreateShader.
    std::cout << "[Shader] Stub: Cargando shaders desde " << vertexPath << " y " << fragmentPath << "\n";
    ID = 0; // ID temporal
}

Shader::~Shader() {
    // glDeleteProgram(ID);
}

void Shader::use() const {
    // glUseProgram(ID);
}

void Shader::setBool(const std::string &name, bool value) const {}
void Shader::setInt(const std::string &name, int value) const {}
void Shader::setFloat(const std::string &name, float value) const {}

void Shader::checkCompileErrors(unsigned int shader, std::string type) const {}