#include "model.hpp"
#include <iostream>

Model::Model(const std::string& path) {
    std::cout << "[Model] Stub: Preparando parseo con Assimp para " << path << "\n";
}

void Model::draw() const {
    // TODO: Iterar vector meshes para enviarlo al Renderer
}