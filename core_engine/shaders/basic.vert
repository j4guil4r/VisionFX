#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // 1. Transformamos la posición al espacio de la cámara (View Space)
    // Multiplicamos projection * view * model * vertice
    vec4 viewPos = view * model * vec4(aPos, 1.0);
    FragPos = vec3(viewPos);

    // 2. Transformamos las normales usando la rotación y escala de view * model
    // Normalizamos para corregir la magnitud producida por la escala de 450
    Normal = normalize(mat3(view * model) * aNormal);

    // 3. Proyección a coordenadas homogéneas de pantalla
    gl_Position = projection * viewPos;
}