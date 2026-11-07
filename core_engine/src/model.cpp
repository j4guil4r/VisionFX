#include "model.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <cctype>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace {
    struct RawMeshData {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        MeshMaterial material;
    };

    std::string toLower(const std::string& str) {
        std::string s = str;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return s;
    }
}

Model::Model(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
        path, 
        aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices
    );

    if(!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        std::cerr << "[ERROR Assimp] Fallo al cargar " << path << ": " << importer.GetErrorString() << "\n";
        return;
    }

    std::vector<RawMeshData> rawMeshes;
    glm::vec3 minBound(std::numeric_limits<float>::max());
    glm::vec3 maxBound(std::numeric_limits<float>::lowest());

    // --- FASE 1: Extraer vértices y determinar materiales iniciales ---
    for(unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        aiMesh* ai_mesh = scene->mMeshes[m];
        RawMeshData raw;

        // Inspección de nombre de malla y material para heurística de visores
        std::string meshName = toLower(ai_mesh->mName.C_Str());
        std::string matName = "";
        aiColor3D diffColor(0.f, 0.f, 0.f);

        if (ai_mesh->mMaterialIndex < scene->mNumMaterials) {
            aiMaterial* mat = scene->mMaterials[ai_mesh->mMaterialIndex];
            aiString name;
            mat->Get(AI_MATKEY_NAME, name);
            matName = toLower(name.C_Str());
            mat->Get(AI_MATKEY_COLOR_DIFFUSE, diffColor);
        }

        // Asignación de material
        bool isVisor = (meshName.find("visor") != std::string::npos ||
                        meshName.find("glass") != std::string::npos ||
                        matName.find("visor") != std::string::npos ||
                        matName.find("glass") != std::string::npos ||
                        matName.find("lente") != std::string::npos);

        if (isVisor) {
            raw.material.objectColor = glm::vec3(0.10f, 0.80f, 0.95f);
            raw.material.shininess = 128.0f;
            raw.material.emission = glm::vec3(0.0f, 0.6f, 0.8f);
        } else if (diffColor.r > 0.01f || diffColor.g > 0.01f || diffColor.b > 0.01f) {
            // Respeta el color base del archivo .mtl si existe
            raw.material.objectColor = glm::vec3(diffColor.r, diffColor.g, diffColor.b);
            raw.material.shininess = 32.0f;
            raw.material.emission = glm::vec3(0.0f);
        } else {
            // Material sólido por defecto
            raw.material.objectColor = glm::vec3(0.85f, 0.45f, 0.15f);
            raw.material.shininess = 16.0f;
            raw.material.emission = glm::vec3(0.0f);
        }

        // Vértices y actualización de la AABB global
        for(unsigned int i = 0; i < ai_mesh->mNumVertices; ++i) {
            Vertex vertex;
            vertex.Position = glm::vec3(ai_mesh->mVertices[i].x, ai_mesh->mVertices[i].y, ai_mesh->mVertices[i].z);

            minBound = glm::min(minBound, vertex.Position);
            maxBound = glm::max(maxBound, vertex.Position);

            if(ai_mesh->HasNormals()) {
                vertex.Normal = glm::vec3(ai_mesh->mNormals[i].x, ai_mesh->mNormals[i].y, ai_mesh->mNormals[i].z);
            } else {
                vertex.Normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            if(ai_mesh->mTextureCoords[0]) {
                vertex.TexCoords = glm::vec2(ai_mesh->mTextureCoords[0][i].x, ai_mesh->mTextureCoords[0][i].y);
            } else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            raw.vertices.push_back(vertex);
        }

        // Índices
        for(unsigned int i = 0; i < ai_mesh->mNumFaces; ++i) {
            aiFace face = ai_mesh->mFaces[i];
            for(unsigned int j = 0; j < face.mNumIndices; ++j)
                raw.indices.push_back(face.mIndices[j]);
        }

        rawMeshes.push_back(raw);
    }

    // Compatibilidad retroactiva: si es tu casco original de 3 mallas sin nombres
    /*if (rawMeshes.size() == 3) {
        rawMeshes[0].material.objectColor = glm::vec3(0.10f, 0.80f, 0.95f);
        rawMeshes[0].material.shininess = 128.0f;
        rawMeshes[0].material.emission = glm::vec3(0.0f, 0.6f, 0.8f);

        rawMeshes[1].material = rawMeshes[0].material;

        rawMeshes[2].material.objectColor = glm::vec3(0.85f, 0.45f, 0.15f);
        rawMeshes[2].material.shininess = 12.0f;
        rawMeshes[2].material.emission = glm::vec3(0.0f);
    }*/

    // --- FASE 2: Normalización Matemática de Geometría (Auto-AABB) ---
    glm::vec3 center = (minBound + maxBound) * 0.5f;
    glm::vec3 size = maxBound - minBound;
    float maxDim = std::max({size.x, size.y, size.z});
    float scaleFactor = (maxDim > 0.0001f) ? (1.0f / maxDim) : 1.0f;

    // --- FASE 3: Centrar, escalar y subir a la GPU ---
    meshes.clear();
    meshes.reserve(rawMeshes.size());
    for(auto& raw : rawMeshes) {
        for(auto& v : raw.vertices) {
            // Centrado geométrico y normalización a caja unitaria [-0.5, 0.5]
            v.Position = (v.Position - center) * scaleFactor;
        }
        meshes.emplace_back(std::move(raw.vertices), std::move(raw.indices), raw.material);
    }

    std::cout << "[Model] Cargado y normalizado: " << path << " (" << meshes.size() 
              << " mallas | Dimensión original: " << maxDim << ")\n";
}

void Model::draw(const Shader& shader) const {
    for(const auto& mesh : meshes) {
        shader.setVec3("objectColor", mesh.material.objectColor.r, mesh.material.objectColor.g, mesh.material.objectColor.b);
        shader.setFloat("shininess", mesh.material.shininess);
        shader.setVec3("emission", mesh.material.emission.r, mesh.material.emission.g, mesh.material.emission.b);
        mesh.draw();
    }
}