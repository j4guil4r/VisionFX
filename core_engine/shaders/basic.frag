#version 330 core
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightDir;
uniform float lightIntensity;
uniform vec3 ambientColor;

// --- Propiedades del Material ---
uniform vec3 objectColor;
uniform float shininess;
uniform vec3 emission;

void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDirNorm = normalize(lightDir);
    vec3 viewDir = normalize(-FragPos);

    // 1. Luz Ambiental
    vec3 ambient = ambientColor * objectColor;

    // 2. Luz Difusa
    float diff = max(dot(norm, lightDirNorm), 0.0);
    diff = max(diff, 0.15);
    vec3 diffuse = diff * lightIntensity * objectColor;

    // 3. Luz Especular (usa el shininess del material)
    vec3 halfwayDir = normalize(lightDirNorm + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    vec3 specular = vec3(1.0, 1.0, 1.0) * spec * (lightIntensity * 0.8);

    // Color final sumando emisión
    vec3 result = ambient + diffuse + specular + emission;
    FragColor = vec4(result, 1.0);
}