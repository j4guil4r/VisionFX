#version 330 core
in vec3 Normal;
out vec4 FragColor;

void main() {
    vec3 color = Normal * 0.5 + 0.5; 
    FragColor = vec4(color, 1.0);
}