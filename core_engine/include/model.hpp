#pragma once
#include "mesh.hpp"
#include <string>
#include <vector>

class Model {
public:
    Model(const std::string& path);
    void draw() const;
private:
    std::vector<Mesh> meshes;
};