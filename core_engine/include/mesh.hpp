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
    unsigned int VAO;
    
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices);
    /**
     * @brief Envía el comando de dibujo a la tarjeta gráfica.
     * Vincula el Vertex Array Object (VAO) actual y despacha los triángulos
     * a la GPU utilizando los índices almacenados en el EBO.
     * @param none
     * @return void
     */
    void draw() const;
private:
    unsigned int VBO, EBO;
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