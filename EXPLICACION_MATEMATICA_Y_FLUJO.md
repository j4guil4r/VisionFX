# Explicación Matemática y Flujo del Pipeline — VisionFX

---

## 1. Ciclo de Vida de un Frame (Pipeline Paso a Paso)

```text
+-----------------------------------------------------------------------------------+
| 1. CÁMARA (app_ui/camera.py :: CameraStream)                                      |
|    Captura asíncrona (1280x720 BGR) + inversión horizontal espejo (cv2.flip).     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. MEDIAPIPE FACE MESH (app_ui/landmark_detector.py :: get_landmarks)             |
|    Inferencia de malla facial sobre frame RGB -> 468 landmarks normalizados.      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. PUNTOS FACIALES (6 Landmarks Anatómicos Rígidos)                               |
|    Índices [1, 199, 33, 263, 61, 291] escalados a coordenadas subpíxel (u_i, v_i) |
|    emparejados con el modelo métrico canónico 3D (FACE_3D en milímetros).         |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. SOLVEPNP + FILTRO ONE-EURO (landmark_detector.py / one_euro_filter.py)         |
|    Levenberg-Marquardt iterativo -> (rvec, tvec) 6-DoF -> Filtrado adaptativo     |
|    en frecuencia para eliminar jitter preservando latencia cero.                  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. CONVERSIÓN DE COORDENADAS (app_ui/main.py)                                     |
|    Rodrigues(rvec) -> R_{3x3} -> Matriz Extrínseca [R|t] -> Inversión de ejes Y,Z |
|    (OpenCV -> OpenGL View Space) + Matriz de Proyección Perspectiva desde K.      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 6. LIGHT ESTIMATOR (core_engine/src/LightEstimator.cpp :: estimate)               |
|    Umbral dinámico (top 20% luminancia) -> Momentos (m00, m10, m01) -> Centroide  |
|    -> Proyección hemisférica en View Space (L_x, L_y, L_z) + Intensidad I_L.      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 7. RENDER EN OPENGL OMNIDIRECCIONAL EN FBO (core_engine/src/renderer.cpp)         |
|    - Pasada 1: Subida de frame BGR (glTexSubImage2D) y dibujo de Fullscreen Quad  |
|                sin Depth Test ni Blending.                                        |
|    - Pasada 2: Limpieza de Z-Buffer, activación de Depth Test y Alpha Blending,   |
|                evaluación de basic.vert y basic.frag (Modo 0, 1 o 2).             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 8. FUSIÓN DE CAPAS EN PANTALLA (get_pixels -> main.py -> gui.py)                  |
|    Lectura directa del FBO en formato GL_BGR (glReadPixels), superposición de     |
|    HUD de modo activo (cv2.putText) y renderizado en QLabel (PyQt6).              |
+-----------------------------------------------------------------------------------+
```

---

## 2. Matemática sin Misterios

### 2.1 Estimación de Pose 6-DoF (Perspective-n-Point)

Dado un conjunto de $n = 6$ puntos 3D canónicos rígidos en el sistema de referencia del rostro $\mathbf{P}_i = [X_i, Y_i, Z_i]^T \in \mathbb{R}^3$ (`FACE_3D`) y sus proyecciones 2D observadas en el plano de imagen $\mathbf{p}_i = [u_i, v_i]^T \in \mathbb{R}^2$, el modelo de cámara *pinhole* relaciona ambos espacios mediante:

$$s_i \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} = \mathbf{K} \left( \mathbf{R}\,\mathbf{P}_i + \mathbf{t} \right), \quad \text{con } \mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

donde:
- $\mathbf{R} = \exp([\mathbf{r}]_\times) \in SO(3)$ es la matriz de rotación ortonormal obtenida del vector eje-ángulo $\mathbf{r} \in \mathbb{R}^3$ (`rvec`) mediante la fórmula de Rodrigues:
  $$\mathbf{R} = \mathbf{I} + \frac{\sin\theta}{\theta}[\mathbf{r}]_\times + \frac{1 - \cos\theta}{\theta^2}[\mathbf{r}]_\times^2, \quad \theta = \|\mathbf{r}\|_2$$
- $\mathbf{t} = [t_x, t_y, t_z]^T \in \mathbb{R}^3$ (`tvec`) es el vector de traslación del origen facial (punta de la nariz) respecto al centro óptico de la cámara.

El algoritmo `cv2.SOLVEPNP_ITERATIVE` (optimización no lineal de **Levenberg-Marquardt**) resuelve los 6 grados de libertad $(\mathbf{r}, \mathbf{t})$ minimizando la suma del **error de reproyección euclídeo cuadrático**:

$$\min_{\mathbf{r}, \mathbf{t}} \sum_{i=1}^{n} \left\| \mathbf{p}_i - \pi\big(\mathbf{K}(\exp([\mathbf{r}]_\times)\mathbf{P}_i + \mathbf{t})\big) \right\|_2^2$$

donde la función de proyección perspectiva deshomogeneiza las coordenadas de cámara $\mathbf{P}_i^c = [X_i^c, Y_i^c, Z_i^c]^T = \mathbf{R}\mathbf{P}_i + \mathbf{t}$:

$$\pi(\mathbf{K}\mathbf{P}_i^c) = \begin{bmatrix} f_x \frac{X_i^c}{Z_i^c} + c_x \\[6pt] f_y \frac{Y_i^c}{Z_i^c} + c_y \end{bmatrix}$$

---

### 2.2 Transformación de Ejes (OpenCV a OpenGL)

OpenCV y OpenGL emplean sistemas de coordenadas de cámara con convenciones de signos opuestas en los ejes vertical y de profundidad:

| Eje Espacial | Convención OpenCV (Camera Space) | Convención OpenGL (View Space) | Operación |
| :--- | :--- | :--- | :--- |
| **Horizontal ($X$)** | $+X$ hacia la derecha | $+X$ hacia la derecha | Se conserva ($+1$) |
| **Vertical ($Y$)** | $+Y$ hacia abajo | $+Y$ hacia arriba | Se invierte ($-1$) |
| **Profundidad ($Z$)** | $+Z$ hacia el frente (escena) | $-Z$ hacia el frente (cámara mira a $-Z$) | Se invierte ($-1$) |

Para pasar la matriz extrínseca de OpenCV $\mathbf{V}_{\text{CV}} \in SE(3)$ al espacio de vista de OpenGL $\mathbf{V}_{\text{GL}}$, se premultiplica por la matriz de reflexión diagonal $\mathbf{M}_{\text{CV}\to\text{GL}} = \text{diag}(1, -1, -1, 1)$:

$$\mathbf{V}_{\text{GL}} = \underbrace{\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}}_{\mathbf{M}_{\text{CV}\to\text{GL}}} \begin{bmatrix} R_{00} & R_{01} & R_{02} & t_x \\ R_{10} & R_{11} & R_{12} & t_y \\ R_{20} & R_{21} & R_{22} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} R_{00} & R_{01} & R_{02} & t_x \\ -R_{10} & -R_{11} & -R_{12} & -t_y \\ -R_{20} & -R_{21} & -R_{22} & -t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

Asimismo, la matriz intrínseca $\mathbf{K}$ se mapea al volumen canónico de recorte (NDC $\in [-1, 1]^3$) con planos $n$ (*near*) y $f$ (*far*) mediante:

$$\mathbf{P}_{\text{GL}} = \begin{bmatrix} \frac{2f_x}{W} & 0 & 1 - \frac{2c_x}{W} & 0 \\[6pt] 0 & \frac{2f_y}{H} & \frac{2c_y}{H} - 1 & 0 \\[6pt] 0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\[6pt] 0 & 0 & -1 & 0 \end{bmatrix}$$

---

### 2.3 Ecuaciones de Sombreado (`core_engine/shaders/basic.frag`)

En el Fragment Shader todos los cálculos se realizan en **View Space** (donde la cámara se sitúa en el origen $\mathbf{0}$). Para cada fragmento con posición interpolada $\mathbf{p}_{\text{frag}}$ (`FragPos`) y normal $\mathbf{n}$ (`Normal`), se definen los vectores unitarios base:

$$\vec{N} = \frac{\mathbf{n}}{\|\mathbf{n}\|_2}, \qquad \vec{L} = \frac{\mathbf{l}_{\text{dir}}}{\|\mathbf{l}_{\text{dir}}\|_2}, \qquad \vec{V} = \frac{-\mathbf{p}_{\text{frag}}}{\|-\mathbf{p}_{\text{frag}}\|_2}, \qquad \vec{H} = \frac{\vec{L} + \vec{V}}{\|\vec{L} + \vec{V}\|_2}$$

#### Modo 1 (`u_RenderMode == 0`) — Realista (Blinn-Phong Reactivo)
Evalúa la reflectancia local combinando componente ambiental, reflexión difusa lambertiana, reflexión especular de Blinn-Phong con exponente de rugosidad $\alpha$ (`shininess`), atenuación cuadrática por distancia normalizada $d$ y emisión propia $I_e$:

$$I = I_a k_a + \Big( k_d \max(\vec{N} \cdot \vec{L}, 0.0) + k_s \max(\vec{N} \cdot \vec{H}, 0.0)^\alpha \Big) \cdot \frac{I_L}{d^2} + I_e$$

donde $I_a$ es `ambientColor`, $k_a = k_d$ es `objectColor`, $k_s$ es el coeficiente especular, $I_L$ es `lightIntensity` estimado en tiempo real por `LightEstimator` y $d = \max(\|\mathbf{p}_{\text{frag}}\|_2 / d_{\text{ref}}, 0.75)$.

#### Modo 2 (`u_RenderMode == 1`) — Cómic / Anime (Cel-Shading / Toon)
Discretiza la transición lumínica continua en 4 bandas planas mediante cuantización escalonada y aplica un contorno negro de silueta cuando la normal es casi ortogonal al vector de visión:

$$NdotL = \max(\vec{N} \cdot \vec{L}, 0.0), \qquad \text{intensity} = \frac{\lfloor NdotL \cdot 4.0 \rfloor}{4.0}$$

$$S_{\text{toon}} = \text{step}\big(0.85, \max(\vec{N} \cdot \vec{H}, 0.0)\big) = \begin{cases} 1.0 & \text{si } \max(\vec{N} \cdot \vec{H}, 0.0) \ge 0.85 \\ 0.0 & \text{en otro caso} \end{cases}$$

$$I_{\text{toon}} = \begin{cases} \mathbf{0} & \text{si } (\vec{N} \cdot \vec{V}) < 0.25 \quad \text{(Borde de Silueta / Rim)} \\[6pt] I_a k_a + k_d \cdot \text{intensity} \cdot I_L + k_s \cdot S_{\text{toon}} \cdot I_L + I_e & \text{si } (\vec{N} \cdot \vec{V}) \ge 0.25 \end{cases}$$

#### Modo 3 (`u_RenderMode == 2`) — Cyberpunk / Holograma (Fresnel + Scanlines)
Combina aproximación de Schlick-Fresnel de orden 3 (bordes hiperbrillantes y centro translúcido) con modulación sinusoidal en espacio de pantalla (`gl_FragCoord.y`) animada en función del tiempo `u_Time` ($t$):

$$\text{fresnel} = \big(1.0 - \max(\vec{N} \cdot \vec{V}, 0.0)\big)^3$$

$$\text{scanline} = 0.5 + 0.5 \sin\big(y_{\text{frag}} \cdot 0.4 + 5.0\,t\big), \quad \text{con } y_{\text{frag}} = \texttt{gl\_FragCoord.y}$$

$$\mathbf{C}_{\text{holo}} = \begin{bmatrix} 0.0 & 0.85 & 1.0 \end{bmatrix}^T$$

$$\texttt{FragColor} = \begin{bmatrix} \mathbf{C}_{\text{holo}} \cdot (\text{fresnel} + 0.2) \cdot \text{scanline} \\[4pt] \text{clamp}(\text{fresnel} \cdot 0.9 + 0.1,\; 0.0,\; 1.0) \end{bmatrix}$$

La componente alfa $\alpha \in [0.1, 1.0]$ se mezcla directamente sobre el *Fullscreen Quad* de la cámara en el FBO mediante la ecuación de composición de Porter-Duff `glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)`:

$$\mathbf{C}_{\text{final}} = \alpha\,\mathbf{C}_{\text{frag}} + (1 - \alpha)\,\mathbf{C}_{\text{fondo}}$$

---

## 3. Mapeo de Módulos

| Archivo | Función/Método | Entrada | Salida | Responsabilidad Matemática |
| :--- | :--- | :--- | :--- | :--- |
| `app_ui/camera.py` | `CameraStream.read` | Stream UVC de hardware | `np.ndarray` $(H \times W \times 3)$ BGR | Adquisición asíncrona de matriz de píxeles cruda sin bloqueo de I/O. |
| `app_ui/landmark_detector.py` | `LandmarkDetector.get_landmarks` | Frame BGR $(H \times W \times 3)$ | Lista de 6 tuplas $(u_i, v_i) \in \mathbb{R}^2$ | Inferencia topológica facial y desnormalización afín a coordenadas subpíxel: $u_i = x_i W, \; v_i = y_i H$. |
| `app_ui/landmark_detector.py` | `LandmarkDetector.estimate_pose` | `points_2d`, $\mathbf{K}$, `dist_coeffs` | Tupla $(\mathbf{r}, \mathbf{t}) \in \mathbb{R}^3 \times \mathbb{R}^3$ | Minimización no lineal Levenberg-Marquardt (`solvePnP`) del error de reproyección respecto a `FACE_3D` con validación $t_z > 0$. |
| `app_ui/one_euro_filter.py` | `OneEuroFilter.filter` | Vector $\mathbf{x}_t$, timestamp $t$ | Vector filtrado $\hat{\mathbf{x}}_t$ | Filtro paso-bajo adaptativo de primer orden con frecuencia de corte dinámica $f_c = f_{\min} + \beta |\dot{\hat{x}}|$ y suavizado exponencial $\alpha = (1 + \frac{1}{2\pi f_c \Delta t})^{-1}$. |
| `app_ui/main.py` | `get_opengl_proj_from_opencv` | $\mathbf{K}$, $W, H, n, f$ | Matriz $\mathbf{P}_{\text{GL}} \in \mathbb{R}^{4\times 4}$ | Mapeo proyectivo de parámetros intrínsecos $(f_x, f_y, c_x, c_y)$ al frustum canónico NDC de OpenGL. |
| `app_ui/main.py` | `VideoPipelineWorker.run` | Stream de cámara, eventos de UI/teclado | Frame compuesto BGR con HUD | Conversión de Rodrigues $\mathbf{r} \to \mathbf{R}$, cambio de base $\mathbf{M}_{\text{CV}\to\text{GL}}$, composición $\mathbf{M} = \mathbf{T}\mathbf{S}\mathbf{R}_{\text{local}}$ y conmutación en caliente `set_render_mode`. |
| `core_engine/src/LightEstimator.cpp` | `LightEstimator::estimate` | `cv::Mat` BGR | `LightEstimate` $(\vec{L}, I_L, I_a)$ | Cálculo de momentos espaciales ponderados $(m_{00}, m_{10}, m_{01})$ del percentil 80 superior de luminancia, proyección hemisférica $L_z = \sqrt{1 - L_x^2 - L_y^2}$ y filtro EMA temporal. |
| `core_engine/src/model.cpp` | `Model::Model` | Ruta de archivo `.obj` | Mallas normalizadas en VRAM (VAO/VBO/EBO) | Cálculo de AABB $[\mathbf{b}_{\min}, \mathbf{b}_{\max}]$, centrado en el origen $\mathbf{c} = \frac{1}{2}(\mathbf{b}_{\min}+\mathbf{b}_{\max})$ y escalado isótropo a caja unitaria $[-0.5, 0.5]^3$. |
| `core_engine/src/renderer.cpp` | `Renderer::render_scene` | `Model`, `modelShader`, `bgShader` | Framebuffer Object (`FBO`) actualizado | Composición en GPU en 2 pasadas, inyección de uniforms dinámicos (`u_RenderMode`, `u_Time`) y mezcla alfa $(\alpha, 1-\alpha)$. |
| `core_engine/shaders/basic.vert` | `main` (Vertex Shader) | `aPos`, `aNormal`, $\mathbf{M}, \mathbf{V}, \mathbf{P}$ | `FragPos`, `Normal`, `gl_Position` | Transformación de vértices y normales a View Space ($\mathbf{V}\mathbf{M}$) y proyección a Clip Space ($\mathbf{P}\mathbf{V}\mathbf{M}$). |
| `core_engine/shaders/basic.frag` | `main` (Fragment Shader) | `FragPos`, `Normal`, `u_RenderMode`, `u_Time` | `FragColor` (RGBA) | Evaluación por píxel de los 3 modelos matemáticos de iluminación: Blinn-Phong con atenuación (0), Cel-Shading + Rim (1) y Fresnel + Scanlines (2). |
| `core_engine/bindings/pybind_wrapper.cpp` | `PYBIND11_MODULE(ar_helmet_core)` | Tensores NumPy y escalares Python | Estructuras C++ / OpenGL | Interoperabilidad *zero-copy* mediante `py::buffer_info` y exposición de `set_render_mode` hacia Python. |

