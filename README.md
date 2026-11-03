# VisionFX: Pipeline de Realidad Aumentada de Alto Rendimiento

> **Pipeline de Realidad Aumentada (AR) a 60-80+ FPS basado en un motor híbrido C++17/OpenGL 3.3 y Python (PyQt6, MediaPipe, OpenCV).**

---

## 📌 Abstract y Resumen Ejecutivo

**VisionFX** es un sistema de Realidad Aumentada facial de alta frecuencia de cuadro que fusiona visión por computador en tiempo real con renderizado gráfico moderno. A diferencia de soluciones basadas en CPU con transferencias pesadas de memoria, VisionFX traslada la composición visual **100% a la GPU** mediante un *Fullscreen Quad* en un único Framebuffer Object (FBO).

### Componentes Clave del Sistema:
- **Seguimiento Facial 6-DoF:** Detección de landmarks faciales 2D con MediaPipe FaceMesh y resolución Perspective-n-Point (`app_ui/landmark_detector.py`) iterativo con respaldo continuo (*fallback* limpio).
- **Estabilización de Pose Adaptativa:** Filtro One-Euro (`app_ui/one_euro_filter.py`) con frecuencias de corte desacopladas para rotación (rad) y traslación (mm), erradicando el temblor (*jitter*) estático sin introducir retraso (*latency*).
- **Estimación de Luz Dinámica en Tiempo Real:** Algoritmo en C++ (`core_engine/src/LightEstimator.cpp`) que calcula la dirección de la fuente lumínica mediante umbralización adaptativa del 20% superior y momentos espaciales proyectados en la semiesfera frontal en View Space.
- **Renderizado Compuesto en GPU:** Motor nativo C++ (`core_engine/src/renderer.cpp`) que dibuja el fondo de cámara 2D y el modelo 3D con sombreado Blinn-Phong en dos pasadas sobre el mismo FBO, eliminando paradas del bus PCIe.
- **Interfaz Reactiva Asíncrona:** UI en PyQt6 con captura desacoplada en hilo dedicado (`app_ui/camera.py`) para alimentar el motor gráfico a máxima tasa de refresco.

---

## 👥 Integrantes del Proyecto

- **Jose Aguilar**
- **Jose Arias Romero**

---

## ⚙️ Requisitos Previos Generales

- **Python:** 3.10 o superior.
- **CMake:** 3.20 o superior.
- **Compilador C++:** C++17 compliant (GCC 11+, Clang 13+, MSVC 19.29+).
- **GPU & Drivers:** Tarjeta gráfica con soporte nativo para **OpenGL 3.3 Core Profile**.
- **Hardware de Entrada:** Cámara web compatible con UVC / OpenCV VideoCapture.

---

## 🚀 Guía de Instalación y Ejecución Paso a Paso

### Opción 1: Linux (Entorno Nativo Recomendado)

Es la plataforma recomendada para desarrollo y despliegue por su baja sobrecarga en contextos GLFW y transferencia OpenGL sin intermediarios.

1. **Instalar paquetes del sistema (Ubuntu/Debian/Pop!_OS):**
   ```bash
   sudo apt update
   sudo apt install -y build-essential cmake libgl1-mesa-dev libglfw3-dev libopencv-dev
   ```

2. **Crear y activar el entorno virtual de Python:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Compilar el motor C++ (`ar_helmet_core`):**
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(nproc)
   ```

4. **Copiar el módulo compilado a la raíz del repositorio:**
   ```bash
   cp build/core_engine/ar_helmet_core*.so .
   ```

5. **Ejecutar la aplicación:**
   ```bash
   python3 -m app_ui.main
   ```

---

### Opción 2: Windows

#### 💡 Disclaimer Destacado: WSL (Recomendado)

> [!IMPORTANT]
> Para garantizar la máxima paridad de rendimiento, evitar discrepancias con dependencias C++ y asegurar compatibilidad nativa con las extensiones de GLFW y OpenGL, se recomienda encarecidamente utilizar **WSL 2 (Ubuntu 22.04 / 24.04)** con soporte GUI nativo (WSLg).  
> Consulta la guía oficial: [Documentación oficial de instalación de WSL](https://learn.microsoft.com/en-us/windows/wsl/install).  
> Tras instalar WSL, sigue simplemente los pasos de la sección de **Linux**.

#### Alternativa: Compilación Nativa en Windows (MSVC)
Si deseas compilar directamente sobre Windows sin WSL:
1. **Requisitos:** Visual Studio 2022 con la carga de trabajo *"Desarrollo para el escritorio con C++"*, CMake y Python 3.10+ en el PATH.
2. **Entorno Python:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Compilar con CMake y MSVC:**
   ```powershell
   cmake -B build -S . -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   copy build\core_engine\Release\ar_helmet_core*.pyd .
   ```
4. **Ejecutar:**
   ```powershell
   python -m app_ui.main
   ```

---

### Opción 3: macOS

1. **Instalar dependencias mediante Homebrew:**
   ```bash
   brew update
   brew install cmake glfw opencv
   ```

2. **Configurar entorno Python:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Compilación del módulo:**
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(sysctl -n hw.ncpu)
   cp build/core_engine/ar_helmet_core*.so .
   ```

4. **Advertencia de Compatibilidad OpenGL:**

> [!WARNING]
> Apple ha marcado la API de OpenGL como obsoleta (*deprecated*) en favor de Metal. No obstante, el sistema operativo mantiene soporte funcional para **OpenGL 3.3 Core Profile**. El motor fuerza explícitamente `GLFW_OPENGL_CORE_PROFILE` y `GLFW_CONTEXT_VERSION_MAJOR 3`, `GLFW_CONTEXT_VERSION_MINOR 3` en `core_engine/src/renderer.cpp`, garantizando compatibilidad tanto en Intel como en Apple Silicon mediante compatibilidad backward.

5. **Ejecutar:**
   ```bash
   python3 -m app_ui.main
   ```

---

## 📚 Documentación Técnica y Fundamentos Matemáticos

Para una sustentación académica exhaustiva y defensa del sistema, consulta los documentos de estudio en la raíz del repositorio:

- 📘 [STUDY_ARQUITECTURA_PIPELINE.md](STUDY_ARQUITECTURA_PIPELINE.md) - **Arquitectura de Pipeline y Flujo 3D de Extremo a Extremo**:
  - Diagrama de bloques y flujo de datos desacoplado.
  - Eliminación de cuellos de botella CPU-GPU (análisis de *pipeline stalls* y solución con *Fullscreen Quad*).
  - Geometría de landmarks, modelo canónico `FACE_3D` y resolución Perspective-n-Point iterativa con Levenberg-Marquardt.
  - Formulación matemática del filtro One-Euro y desacople cinemático de rotación/traslación.
  - Transformaciones proyectivas entre espacios de coordenadas (OpenCV $\to$ OpenGL View Space $\to$ Clip Space).

- 💡 [STUDY_LIGHT_ESTIMATOR.md](STUDY_LIGHT_ESTIMATOR.md) - **Estimación de Luz Dinámica en Tiempo Real**:
  - Demostración analítica de la falla del centroide simple en escenas con fondo iluminado.
  - Algoritmo de umbral dinámico adaptativo del 20% superior y extracción de momentos de imagen $m_{00}, m_{10}, m_{01}$.
  - Proyección continua en la semiesfera frontal en View Space ($Z = \sqrt{1 - r^2}$).
  - Filtro temporal asimétrico EMA para respuesta instantánea de iluminación.
  - Formulación del shader Blinn-Phong modificado y control de sombras especulares.
