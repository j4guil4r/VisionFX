import sys
import time
import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

import ar_helmet_core

try:
    from app_ui.gui import MainWindow
    from app_ui.landmark_detector import LandmarkDetector
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
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 60)
        return cap

    def run(self):
        print("[VideoWorker] Levantando motor C++ Offscreen...")
        motor = ar_helmet_core.Renderer(1280, 720)
        shader = ar_helmet_core.Shader("core_engine/shaders/basic.vert", "core_engine/shaders/basic.frag")
        modelo = ar_helmet_core.Model("model.obj")
        
        # Pre-calculamos la proyección y modelo
        proj_mat = create_projection_matrix(45.0, 1280/720, 0.1, 1000.0)
        
        # Modelo
        model_mat = np.eye(4, dtype=np.float32)
        escala = 3.0
        model_mat[0, 0] = -escala  # Escala y rotación 180° en Y simplificada
        model_mat[1, 1] = escala
        model_mat[2, 2] = -escala

        cap = self._open_camera(self._camera_index)
        prev_time = time.time()
        fps = 0.0

        while True:
            self._mutex.lock()
            if not self._running:
                self._mutex.unlock()
                break

            # Cambio de cámara en caliente
            if self._pending_camera_index != self._camera_index:
                self._camera_index = self._pending_camera_index
                if cap is not None and cap.isOpened():
                    cap.release()
                cap = self._open_camera(self._camera_index)

            mask_on = self._mask_enabled
            self._mutex.unlock()

            if cap is None or not cap.isOpened():
                # Frame placeholder si la cámara seleccionada no está disponible
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

            ret, frame = cap.read()
            if not ret:
                self.msleep(10)
                continue

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
                if points_2d:
                    pose = self._detector.estimate_pose(points_2d, self._cam_matrix, self._dist_coeffs)
                    if pose:
                        rvec, tvec = pose
                        
                        # -- DEBUG VISUAL: Dibujamos los ejes 3D de OpenCV --
                        proj_axes, _ = cv2.projectPoints(AXES_3D, rvec, tvec, self._cam_matrix, self._dist_coeffs)
                        pts = proj_axes.reshape(-1, 2).astype(int)
                        origin = tuple(pts[0])
                        cv2.line(frame, origin, tuple(pts[1]), (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.line(frame, origin, tuple(pts[2]), (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.line(frame, origin, tuple(pts[3]), (255, 0, 0), 3, cv2.LINE_AA)

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

                        # Calculamos proyeccion exacta a partir del feed real
                        # Aumentamos el plano lejano a 10000 para que no la recorte
                        proj_mat = get_opengl_proj_from_opencv(self._cam_matrix, w, h, 0.1, 10000.0)
                        escala_real = 450.0 
                        model_mat = np.eye(4, dtype=np.float32)
                        model_mat[0, 0] = escala_real 
                        model_mat[1, 1] = -escala_real
                        model_mat[2, 2] = escala_real
                        
                        # Traslación de compensación (Offset)
                        # AJUSTAR A CONVENIENCIA (jugar con estos valores)
                        offset_y = -50
                        offset_z = 0.0
                        
                        model_mat[1, 3] = offset_y
                        model_mat[2, 3] = offset_z

                        shader.use()
                        shader.set_mat4("projection", proj_mat.flatten('F').tolist())
                        shader.set_mat4("view", view_mat.flatten('F').tolist())
                        shader.set_mat4("model", model_mat.flatten('F').tolist())
                        
                        motor.render(modelo)
                        fbo_pixels = motor.get_frame()
                        
                        mask_frame = cv2.flip(fbo_pixels, 0)
                        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_RGB2BGR)
                        
                        alpha = np.any(mask_frame != [0, 0, 0], axis=-1)
                        frame[alpha] = mask_frame[alpha]

            # Cálculo y suavizado de FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            self.frame_ready.emit(frame)
            self.fps_updated.emit(fps)

        if cap is not None and cap.isOpened():
            cap.release()

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
