#version 330 core
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightDir;        // Dirección de la luz en View Space
uniform float lightIntensity; // Brillo [0.0 - 1.0]
uniform vec3 ambientColor;    // Luz ambiente tenue
uniform vec3 objectColor;     // Color base del casco

void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDirNorm = normalize(lightDir);

    // Vector hacia el observador (en View Space la cámara siempre está en 0,0,0)
    vec3 viewDir = normalize(-FragPos);

    // 1. Componente Ambiental
    vec3 ambient = ambientColor * objectColor;

    // 2. Componente Difusa (Lambert)
    // Usamos max para incidencia frontal
    float diff = max(dot(norm, lightDirNorm), 0.0);
    // Si la cara está opuesta, le damos un 15% de luz secundaria para no perder aristas
    diff = max(diff, 0.15); 
    vec3 diffuse = diff * lightIntensity * objectColor;

    // 3. Componente Especular (Brillo blanco Blinn-Phong)
    vec3 halfwayDir = normalize(lightDirNorm + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), 16.0);
    vec3 specular = vec3(1.0, 1.0, 1.0) * spec * (lightIntensity * 0.8);

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}