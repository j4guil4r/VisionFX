#pragma once
#include "mesh.hpp"
#include <string>
#include <vector>
#include "shader.hpp"

class Model {
public:
    Model(const std::string& path);
    void draw(const Shader& shader) const;
    
private:
    std::vector<Mesh> meshes;
};