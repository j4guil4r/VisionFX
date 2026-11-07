#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>


struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

struct MeshMaterial {
    glm::vec3 objectColor{0.85f, 0.45f, 0.15f};
    float shininess{24.0f};
    glm::vec3 emission{0.0f, 0.0f, 0.0f};
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    unsigned int VAO{0};
    MeshMaterial material;
    
    Mesh(std::vector<Vertex> in_vertices, std::vector<unsigned int> in_indices, MeshMaterial in_mat = MeshMaterial());
    /**
     * @brief Envía el comando de dibujo a la tarjeta gráfica.
     * Vincula el Vertex Array Object (VAO) actual y despacha los triángulos
     * a la GPU utilizando los índices almacenados en el EBO.
     * @param none
     * @return void
     */
    void draw() const;
private:
    unsigned int VBO{0}, EBO{0};
    /**
     * @brief Reserva y configura la memoria en la VRAM de la GPU.
     * Genera los buffers (VAO, VBO, EBO), transfiere los vectores de vértices
     * e índices de la memoria RAM a la GPU, y define el layout de los atributos
     * (posiciones, normales y coordenadas de textura) para el shader.
     * @param none
     * @return void
     */
    void setupMesh();
};