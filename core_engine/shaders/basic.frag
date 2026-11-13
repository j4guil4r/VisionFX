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
        // Modo 2: Cómic / Anime (Cel-Shading calibrado para colores claros y oscuros)
        if (dot(N, V) < 0.22) {
            FragColor = vec4(vec3(0.02), 1.0);
        } else {
            float halfLambert = dot(N, L) * 0.5 + 0.5;
            
            // 3 bandas bien diferenciadas: 0.0 (sombra), 0.5 (medio), 1.0 (luz)
            float band = floor(halfLambert * 2.99) / 2.0;
            // Sombra en 0.36 y luz máxima en 0.92 evita que el amarillo sature a plano
            float toonFactor = mix(0.36, 0.92, band);

            float NdotL = max(dot(N, L), 0.0);
            float specRaw = pow(max(dot(N, H), 0.0), max(shininess, 36.0));
            float spec = (NdotL > 0.1) ? step(0.55, specRaw) : 0.0;

            float safeIntensity = clamp(lightIntensity, 0.80, 1.05);

            vec3 diffuse = objectColor * toonFactor * safeIntensity;
            vec3 specular = mix(objectColor, vec3(1.0), 0.8) * spec * 0.40;

            vec3 result = diffuse + specular + emission;
            FragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
        }
    } else if (u_RenderMode == 2) {
        // Modo 3: Cyberpunk / Holograma (Fresnel + Scanlines)
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        vec3 holoColor = vec3(0.0, 0.85, 1.0);
        float scanline = 0.5 + 0.5 * sin(gl_FragCoord.y * 0.4 + u_Time * 5.0);
        FragColor = vec4(holoColor * (fresnel + 0.2) * scanline, clamp(fresnel * 0.9 + 0.1, 0.0, 1.0));
    } else {
        // Modo 1 (u_RenderMode == 0): Realista con Tone Mapping suave y Alto Reflejo
        vec3 ambient = max(ambientColor, vec3(0.16)) * objectColor;

        float NdotL = max(dot(N, L), 0.0);
        float wrapDiff = dot(N, L) * 0.5 + 0.5;
        float directionalDiff = mix(pow(NdotL, 1.15), wrapDiff, 0.20);
        // Ganancia difusa en 0.92 preserva el degradado en colores brillantes como el amarillo
        vec3 diffuse = objectColor * directionalDiff * 0.92;

        // Doble lóbulo especular para reflejos con linterna
        float specBroad = 0.0;
        float specSharp = 0.0;
        if (NdotL > 0.0) {
            float NdotH = max(dot(N, H), 0.0);
            specBroad = pow(NdotH, 10.0) * 0.40;
            specSharp = pow(NdotH, max(shininess, 48.0)) * 1.45;
        }

        vec3 specColor = mix(objectColor, vec3(1.0), 0.75);
        vec3 specular = specColor * (specBroad + specSharp);

        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        vec3 rimSpec = vec3(1.0) * fresnel * NdotL * 0.45;

        float lightPower = pow(clamp(lightIntensity, 0.65, 1.35), 1.20) * 1.15;

        vec3 rawColor = ambient + (diffuse + specular + rimSpec) * lightPower + emission;
        // Compresión suave (Tone Mapping) para no quemar highlights
        vec3 mapped = rawColor / (1.0 + rawColor * 0.22);
        FragColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);
    }
}