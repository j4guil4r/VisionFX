#version 330 core
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightDir;
uniform float lightIntensity;
uniform vec3 ambientColor;

uniform vec3 objectColor;
uniform float shininess;
uniform vec3 emission;

void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDirNorm = normalize(lightDir);
    vec3 viewDir = normalize(-FragPos);

    // Luz Ambiental
    vec3 ambient = ambientColor * objectColor;

    // Luz Difusa
    float diff = max(dot(norm, lightDirNorm), 0.0);
    vec3 diffuse = diff * lightIntensity * objectColor;

    // Luz Especular
    vec3 halfwayDir = normalize(lightDirNorm + viewDir);
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    }
    vec3 specular = vec3(1.0, 1.0, 1.0) * spec * (lightIntensity * 0.9);

    
    vec3 result = ambient + diffuse + specular + emission;
    FragColor = vec4(result, 1.0);
}