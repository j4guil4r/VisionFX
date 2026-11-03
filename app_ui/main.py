import sys
import time
import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
from app_ui.one_euro_filter import OneEuroFilter

import ar_helmet_core
import platform

try:
    from app_ui.gui import MainWindow
    from app_ui.landmark_detector import LandmarkDetector
    from app_ui.camera import CameraStream
except ImportError:
    from gui import MainWindow
    from landmark_detector import LandmarkDetector

# Ejes 3D de referencia canónica (X: Rojo, Y: Verde, Z: Azul)
AXES_3D = np.array([
    [0.0, 0.0, 0.0],
    [60.0, 0.0, 0.0],
    [0.0, -60.0, 0.0],
    [0.0, 0.0, -60.0]
], dtype=np.float64)


def get_opengl_proj_from_opencv(cam_mat, width, height, near, far):
    """Convierte la matriz intrínseca de OpenCV en una Matriz de Proyección de OpenGL"""
    fx = cam_mat[0, 0]
    fy = cam_mat[1, 1]
    cx = cam_mat[0, 2]
    cy = cam_mat[1, 2]
    
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height
    proj[2, 0] = 1.0 - (2.0 * cx / width)
    proj[2, 1] = (2.0 * cy / height) - 1.0
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0
    return proj

def create_projection_matrix(fov, aspect_ratio, near, far):
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect_ratio
    mat[1, 1] = f
    mat[2, 2] = -(far + near) / (far - near)
    mat[2, 3] = -(2.0 * far * near) / (far - near)
    mat[3, 2] = -1.0
    return mat

class VideoPipelineWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self._mutex = QMutex()
        self._running = True
        self._camera_index = camera_index
        self._pending_camera_index = camera_index
        self._mask_enabled = True

        self._detector = LandmarkDetector()
        self._cam_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def set_mask_enabled(self, enabled: bool):
        self._mutex.lock()
        self._mask_enabled = enabled
        self._mutex.unlock()

    def change_camera(self, index: int):
        self._mutex.lock()
        self._pending_camera_index = index
        self._mutex.unlock()

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()
        self.wait()

    def _open_camera(self, index: int):
        os = platform.system()
        if os == "Linux":
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        elif os == "Windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index) # macOS

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        print("[VideoWorker] Levantando motor C++ Offscreen...")
        light_estimator = ar_helmet_core.LightEstimator()
        motor = ar_helmet_core.Renderer(1280, 720)
        # Shaders
        shader = ar_helmet_core.Shader("core_engine/shaders/basic.vert", "core_engine/shaders/basic.frag")
        bg_shader = ar_helmet_core.Shader("core_engine/shaders/camera_bg.vert", "core_engine/shaders/camera_bg.frag")
        modelo = ar_helmet_core.Model("model.obj")

        # Filtros de pose para estabilizar rotación y traslación
        filter_rvec = OneEuroFilter(min_cutoff=1.2, beta=0.8, d_cutoff=1.0)
        # Traslación (mm): valores estándar
        filter_tvec = OneEuroFilter(min_cutoff=0.8, beta=0.03, d_cutoff=1.0)

        last_rvec = None
        last_tvec = None
        lost_frames = 0
        MAX_LOST_FRAMES = 4

        stream = CameraStream(self._camera_index).start()
        
        # Pre-calculamos la proyección y modelo
        #proj_mat = create_projection_matrix(45.0, 1280/720, 0.1, 1000.0)
        
        # Modelo
        #model_mat = np.eye(4, dtype=np.float32)
        #escala = 3.0
        #model_mat[0, 0] = -escala  # Escala y rotación 180° en Y simplificada
        #model_mat[1, 1] = escala
        #model_mat[2, 2] = -escala

        #cap = self._open_camera(self._camera_index)
        prev_time = time.time()
        fps = 0.0

        #frame_count = 0

        while True:
            self._mutex.lock()
            if not self._running:
                self._mutex.unlock()
                break

            # Cambio de cámara en caliente
            if self._pending_camera_index != self._camera_index:
                self._camera_index = self._pending_camera_index
                #if cap is not None and cap.isOpened():
                #    cap.release()
                #cap = self._open_camera(self._camera_index)
                stream.change_camera(self._camera_index)

            mask_on = self._mask_enabled
            self._mutex.unlock()

            if not stream.is_opened():
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    f"Camara {self._camera_index} no disponible",
                    (380, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (100, 100, 255),
                    2,
                    cv2.LINE_AA,
                )
                self.frame_ready.emit(placeholder)
                self.msleep(100)
                continue

            #t0 = time.perf_counter()
            ret, frame = stream.read()
            if not ret or frame is None:
                self.msleep(2)
                continue
            #t1 = time.perf_counter()

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Inicialización perezosa de calibración básica
            if self._cam_matrix is None:
                self._cam_matrix = np.array([
                    [w, 0, w / 2.0],
                    [0, w, h / 2.0],
                    [0, 0, 1.0]
                ], dtype=np.float64)

            # Procesamiento de máscara y pose 3D si está activo
            if mask_on:
                points_2d = self._detector.get_landmarks(frame)
                pose = None
                
                if points_2d:
                    pose = self._detector.estimate_pose(points_2d, self._cam_matrix, self._dist_coeffs)

                if pose:
                    rvec_raw, tvec_raw = pose
                    lost_frames = 0
                    t_now = time.perf_counter()

                    # Filtrado adaptativo
                    rvec = filter_rvec.filter(rvec_raw, t_now)
                    tvec = filter_tvec.filter(tvec_raw, t_now)
                    last_rvec = rvec.copy()
                    last_tvec = tvec.copy()

                elif last_rvec is not None and lost_frames < MAX_LOST_FRAMES:
                    # Si perdimos la cara por movimiento rápido, usamos la última pose conocida
                    # Evita que el casco desaparezca o salte al centro de la pantalla
                    lost_frames += 1
                    rvec = last_rvec
                    tvec = last_tvec
                else:
                    # Pérdida definitiva: reseteamos filtros y memoria de solvePnP
                    lost_frames += 1
                    last_rvec = None
                    last_tvec = None
                    filter_rvec.reset()
                    filter_tvec.reset()
                    self._detector.reset_pose()
                    rvec, tvec = None, None

                # Renderizar solo si tenemos una pose válida
                if rvec is not None and tvec is not None:
                    # -- PIPELINE OPENGL -- #
                    rmat, _ = cv2.Rodrigues(rvec)
                    
                    view_mat = np.eye(4, dtype=np.float32)
                    view_mat[:3, :3] = rmat
                    view_mat[:3, 3] = tvec.flatten()
                    
                    cv_to_gl = np.array([
                        [ 1,  0,  0,  0],
                        [ 0, -1,  0,  0],
                        [ 0,  0, -1,  0],
                        [ 0,  0,  0,  1]
                    ], dtype=np.float32)
                    view_mat = cv_to_gl @ view_mat

                    proj_mat = get_opengl_proj_from_opencv(self._cam_matrix, w, h, 0.1, 10000.0)
                    escala_real = 450.0 
                    model_mat = np.eye(4, dtype=np.float32)
                    model_mat[0, 0] = escala_real 
                    model_mat[1, 1] = -escala_real
                    model_mat[2, 2] = escala_real
                    
                    offset_y = -35
                    offset_z = 15.0
                    model_mat[1, 3] = offset_y
                    model_mat[2, 3] = offset_z
                    
                    light_dir, intensity, ambient_col = light_estimator.estimate(frame)

                    shader.use()
                    shader.set_mat4("projection", proj_mat.flatten('F').tolist())
                    shader.set_mat4("view", view_mat.flatten('F').tolist())
                    shader.set_mat4("model", model_mat.flatten('F').tolist())

                    lx = float(light_dir[0])
                    ly = float(light_dir[1])
                    lz = float(light_dir[2])

                    shader.set_vec3("lightDir", lx, ly, lz)
                    shader.set_float("lightIntensity", float(intensity))
                    shader.set_vec3("ambientColor", 0.12, 0.12, 0.12)
                                            
                    motor.update_camera_frame(frame)
                    motor.render_scene(modelo, shader, bg_shader)
                    frame = cv2.flip(motor.get_frame(), 0)

            #t3 = time.perf_counter()
            # Cálculo y suavizado de FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            self.frame_ready.emit(frame)
            self.fps_updated.emit(fps)

            #frame_count+=1
            #if frame_count % 30 == 0:
            #    print(f"[Profiling] Cam: {(t1-t0)*1000:.1f}ms | MediaPipe: {(t2-t1)*1000:.1f}ms | Render+Blend: {(t3-t2)*1000:.1f}ms")

        stream.stop()
        del shader
        del motor


def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    worker = VideoPipelineWorker(camera_index=0)

    # Conexión de señales y slots desacoplados
    worker.frame_ready.connect(window.update_frame)
    worker.fps_updated.connect(window.update_fps)
    window.mask_toggled.connect(worker.set_mask_enabled)
    window.camera_changed.connect(worker.change_camera)
    window.window_closed.connect(worker.stop)

    worker.start()
    window.show()

    exit_code = app.exec()
    worker.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
