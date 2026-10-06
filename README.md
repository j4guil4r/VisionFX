# VisionFX: Sistema de Renderizado de Mallas 3D Reactivas en Tiempo Real

Curso: CS4016 — Computación Gráfica  
Integrantes: José Arias Romero, Jose Aguilar Millones  
Estado: Fase 2 (En desarrollo activo)

Aplicación de Realidad Aumentada que renderiza modelos 3D sobre el rostro del usuario en tiempo real con oclusión geométrica e iluminación reactiva al entorno.

## 🏛 Arquitectura Híbrida

El sistema utiliza una arquitectura de dos capas enlazadas por memoria compartida:

1.  **Capa de Visión e Interfaz (Python):** Utiliza OpenCV y MediaPipe para capturar el video, extraer 6 puntos faciales canónicos y calcular la pose de la cabeza (6-DoF) resolviendo el problema Perspective-n-Point (PnP).
2.  **Motor de Renderizado Nativo (C++):** Construido sobre OpenGL 3.3 Core Profile. Gestiona la carga de geometría con Assimp, el contexto gráfico oculto con GLFW, matemáticas con GLM y el pipeline de renderizado de 3 pasadas.
3.  **Puente (pybind11):** Compila el motor C++ como un módulo `.so` (Linux) o `.pyd` (Windows) importable nativamente desde Python sin copias de memoria innecesarias.

## ⚙️ Requisitos Previos

*   **Python:** 3.10+
*   **CMake:** 3.20+
*   **Compilador C++:** 
    *   *Linux:* GCC 12+ o Clang 15+
    *   *Windows:* MSVC 17+ (Visual Studio 2022) o MinGW-w64.
*   **Librerías del Sistema (Solo Linux):** Puede requerir paquetes de desarrollo gráficos (ej. `libgl1-mesa-dev`, `libxrandr-dev`, `libxi-dev`).

## 🛠 Instrucciones de Construcción (Build)

### 1. Clonar el repositorio e instalar dependencias de Python

```bash
git clone https://github.com/j4guil4r/VisionFX.git
cd VisionFX
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Compilar el Motor Central (Core Engine)
La compilación descargará automáticamente pybind11, GLFW, GLM y Assimp (optimizado exclusivamente para archivos .obj).
En Linux (Pop!_OS / Ubuntu):

```bash 
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cp build/core_engine/ar_helmet_core*.so .
```
En Windows (Visual Studio / MSVC):
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
copy build\core_engine\Release\ar_helmet_core*.pyd .
```

En Windows (MinGW):
```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
cp build/core_engine/ar_helmet_core*.pyd .
```


### 3. Verificar el Enlace
Abre una terminal interactiva de Python y ejecuta:
```python
import ar_helmet_core
modelo = ar_helmet_core.Model("fantasma.obj")
# Debería mostrar un error controlado de Assimp indicando que el archivo no existe.
```


### 🚀 Ejecución del Prototipo Actual
Actualmente, el proyecto se encuentra validando el pipeline de visión. Para ejecutar la demostración del seguimiento 3D:
```python
python app_ui/camera.py
```
(Presiona 'Q' o 'ESC' para salir de la ventana).