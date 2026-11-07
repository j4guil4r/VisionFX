#pragma once
#include "mesh.hpp"
#include <string>
#include <vector>
#include "shader.hpp"

class Model {
public:
    explicit Model(const std::string& path);
    void draw(const Shader& shader) const;

    const std::vector<Mesh>& getMeshes() const { return meshes; }
    
private:
    std::vector<Mesh> meshes;
};