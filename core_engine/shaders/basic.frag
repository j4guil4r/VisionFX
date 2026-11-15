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
uniform float u_Surge;

void main() {
    vec3 N = (length(Normal) > 1e-4) ? normalize(Normal) : vec3(0.0, 0.0, 1.0);
    vec3 L = (length(lightDir) > 1e-4) ? normalize(lightDir) : vec3(0.0, 0.35, 0.93);
    vec3 V = (length(FragPos) > 1e-4) ? normalize(-FragPos) : vec3(0.0, 0.0, 1.0);
    vec3 LV = L + V;
    vec3 H = (length(LV) > 1e-4) ? normalize(LV) : vec3(0.0, 0.0, 1.0);

    // Factor de sobrecarga (gesto de apertura de boca)
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float oneMinusNdotV = max(1.0 - NdotV, 1e-4);

    float surge = clamp(u_Surge, 0.0, 1.0);
    float surgeWave = 0.5 + 0.5 * sin(FragPos.y * 0.09 - u_Time * 18.0);
    float surgeBands = smoothstep(0.60, 0.95, surgeWave);
    float surgePulse = 0.65 + 0.35 * sin(u_Time * 16.0);
    float surgeRim = pow(oneMinusNdotV, 2.0);
    vec3 plasmaColor = mix(vec3(0.0, 0.95, 1.0), vec3(0.55, 0.20, 1.0), 0.5 + 0.5 * sin(u_Time * 10.0));

    if (u_RenderMode == 1) {
        // Modo 2: Cómic / Anime (Cel-Shading calibrado para colores claros y oscuros)
        if (dot(N, V) < 0.22) {
            vec3 inkOutline = mix(vec3(0.02), vec3(0.0, 0.95, 1.0), surge * surgePulse);
            FragColor = vec4(inkOutline, 1.0);
        } else {
            float halfLambert = dot(N, L) * 0.5 + 0.5;
            
            // 3 bandas bien diferenciadas: 0.0 (sombra), 0.5 (medio), 1.0 (luz)
            float band = floor(halfLambert * 2.99) / 2.0;
            // Sombra en 0.36 y luz máxima en 0.92 evita que el amarillo sature a plano
            float toonFactor = mix(0.36, 0.92, band);

            float NdotL = max(dot(N, L), 0.0);
            float specRaw = pow(max(dot(N, H), 1e-4), max(shininess, 36.0));
            float spec = (NdotL > 0.1) ? step(0.55, specRaw) : 0.0;

            float safeIntensity = clamp(lightIntensity, 0.80, 1.05);

            vec3 diffuse = objectColor * toonFactor * safeIntensity;
            vec3 specular = mix(objectColor, vec3(1.0), 0.8) * spec * 0.40;

            vec3 result = diffuse + specular + emission;
            if (surge > 0.01) {
                float toonBandSurge = step(0.70, surgeWave);
                result = mix(result, plasmaColor, surge * (0.35 * surgeRim + 0.45 * toonBandSurge));
            }
            FragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
        }
    } else if (u_RenderMode == 2) {
        // Modo 3: Cyberpunk / Holograma (Fresnel + Scanlines)
        float fresnel = pow(oneMinusNdotV, 3.0);
        vec3 baseHolo = vec3(0.0, 0.85, 1.0);
        vec3 surgeHolo = mix(vec3(1.0, 0.15, 0.85), vec3(0.0, 1.0, 0.95), surgeBands);
        vec3 holoColor = mix(baseHolo, surgeHolo, surge);
        float speed = mix(5.0, 22.0, surge);
        float scanline = 0.5 + 0.5 * sin(gl_FragCoord.y * 0.4 + u_Time * speed);
        float alpha = clamp(fresnel * 0.9 + 0.1 + surge * 0.25, 0.0, 1.0);
        FragColor = vec4(holoColor * (fresnel + 0.2 + surge * 0.45 * surgePulse) * scanline, alpha);
    } else {
        // Modo 1 (u_RenderMode == 0): Realista con Tone Mapping suave y Alto Reflejo
        vec3 ambient = max(ambientColor, vec3(0.16)) * objectColor;

        float NdotL = max(dot(N, L), 0.0);
        float wrapDiff = dot(N, L) * 0.5 + 0.5;
        float directionalDiff = mix(pow(max(NdotL, 1e-4), 1.15), wrapDiff, 0.20);
        // Ganancia difusa en 0.92 preserva el degradado en colores brillantes como el amarillo
        vec3 diffuse = objectColor * directionalDiff * 0.92;

        // Doble lóbulo especular para reflejos con linterna
        float specBroad = 0.0;
        float specSharp = 0.0;
        if (NdotL > 0.0) {
            float NdotH = max(dot(N, H), 1e-4);
            specBroad = pow(NdotH, 10.0) * 0.40;
            specSharp = pow(NdotH, max(shininess, 48.0)) * 1.45;
        }

        vec3 specColor = mix(objectColor, vec3(1.0), 0.75);
        vec3 specular = specColor * (specBroad + specSharp);

        float fresnel = pow(oneMinusNdotV, 3.0);
        vec3 rimSpec = vec3(1.0) * fresnel * NdotL * 0.45;

        float lightPower = pow(clamp(lightIntensity, 0.65, 1.35), 1.20) * 1.15;

        vec3 rawColor = ambient + (diffuse + specular + rimSpec) * lightPower + emission;
        // Compresión suave (Tone Mapping) para no quemar highlights
        vec3 mapped = rawColor / (1.0 + rawColor * 0.22);

        if (surge > 0.01) {
            float visorBoost = (shininess > 64.0) ? 0.65 : 0.25;
            vec3 plasmaGlow = plasmaColor * (surgeRim * 0.75 + surgeBands * 0.55 + visorBoost * surgePulse);
            mapped = mix(mapped, clamp(mapped + plasmaGlow, 0.0, 1.0), surge * 0.85);
        }
        FragColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);
    }
}