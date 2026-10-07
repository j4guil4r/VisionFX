#include "model.hpp"
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

Mesh processMesh(aiMesh *mesh, const aiScene *scene){
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    // Recorrer cada vértice de la malla
    for(unsigned int i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        
        // Posiciones 3D
        vertex.Position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        
        // Normales (Cruciales para el cálculo de luz reactiva de la Fase 4)
        if (mesh->HasNormals()) {
            vertex.Normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        } else {
            vertex.Normal = glm::vec3(0.0f);
        }
        
        // Coordenadas UV (Mapeo de texturas)
        if(mesh->mTextureCoords[0]) {
            vertex.TexCoords = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
        } else {
            vertex.TexCoords = glm::vec2(0.0f, 0.0f);
        }
        
        vertices.push_back(vertex);
    }
    
    // Recorrer las caras (triángulos) y extraer los índices
    for(unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for(unsigned int j = 0; j < face.mNumIndices; j++)
            indices.push_back(face.mIndices[j]);
    }
    
    return Mesh(vertices, indices);
}

void processNode(aiNode *node, const aiScene *scene, std::vector<Mesh>& meshes) {
    for(unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));
    }
    for(unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene, meshes);
    }
}

Model::Model(const std::string& path) {
    Assimp::Importer importer;
    // aiProcess_Triangulate fuerza que todos los polígonos sean triángulos
    // aiProcess_GenSmoothNormals calcula normales si el .obj no las trae
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);

    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "[ERROR Assimp] No se pudo cargar " << path << ": " << importer.GetErrorString() << "\n";
        return;
    }

    processNode(scene->mRootNode, scene, meshes);
    std::cout << "[Model] Cargado exitosamente: " << path << " (" << meshes.size() << " mallas, " 
              << meshes[0].vertices.size() << " vertices)\n";
}

void Model::draw() const {
    for(unsigned int i = 0; i < meshes.size(); ++i){
        meshes[i].draw();
    }
}