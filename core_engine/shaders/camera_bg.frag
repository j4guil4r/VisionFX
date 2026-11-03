#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D cameraTexture;

void main() {
    FragColor = texture(cameraTexture, TexCoords);
}