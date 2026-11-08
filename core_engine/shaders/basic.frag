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

uniform int u_RenderMode;
uniform float u_Time;

void main() {
    vec3 N = normalize(Normal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(-FragPos);
    vec3 H = normalize(L + V);

    if (u_RenderMode == 1) {
        // Modo 2: Cómic / Anime (Cel-Shading / Toon)
        if (dot(N, V) < 0.25) {
            FragColor = vec4(vec3(0.0), 1.0);
        } else {
            float NdotL = max(dot(N, L), 0.0);
            float intensity = floor(NdotL * 4.0) / 4.0;
            float spec = (NdotL > 0.0) ? step(0.85, max(dot(N, H), 0.0)) : 0.0;

            vec3 ambient = ambientColor * objectColor;
            vec3 diffuse = intensity * lightIntensity * objectColor;
            vec3 specular = vec3(1.0, 1.0, 1.0) * spec * (lightIntensity * 0.9);

            vec3 result = ambient + diffuse + specular + emission;
            FragColor = vec4(result, 1.0);
        }
    } else if (u_RenderMode == 2) {
        // Modo 3: Cyberpunk / Holograma (Fresnel + Scanlines)
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        vec3 holoColor = vec3(0.0, 0.85, 1.0);
        float scanline = 0.5 + 0.5 * sin(gl_FragCoord.y * 0.4 + u_Time * 5.0);
        FragColor = vec4(holoColor * (fresnel + 0.2) * scanline, clamp(fresnel * 0.9 + 0.1, 0.0, 1.0));
    } else {
        // Modo 1 (u_RenderMode == 0): Realista (Blinn-Phong reactivo con atenuación)
        float d = max(length(FragPos) / 600.0, 0.75);
        float attenuation = 1.0 / (d * d);

        // Luz Ambiental: I_a * k_a
        vec3 ambient = ambientColor * objectColor;

        // Luz Difusa: k_d * max(N · L, 0.0)
        float NdotL = max(dot(N, L), 0.0);
        vec3 diffuse = objectColor * NdotL;

        // Luz Especular: k_s * max(N · H, 0.0)^alpha
        float spec = 0.0;
        if (NdotL > 0.0) {
            spec = pow(max(dot(N, H), 0.0), shininess);
        }
        vec3 specular = vec3(0.9, 0.9, 0.9) * spec;

        vec3 result = ambient + (diffuse + specular) * (lightIntensity * attenuation) + emission;
        FragColor = vec4(result, 1.0);
    }
}