#pragma once
#include <glm/glm.hpp>
#include <vector>

// Estructura de datos por vertice
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
        : vertices(vertices), indices(indices) {}
    
    // TODO: configuracion de hardware (VAO, VBO, EBO)
};