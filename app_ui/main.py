import os
import sys
import time
import cv2
from pathlib import Path
import numpy as np
from PyQt6.QtCore import QMutex, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _candidate in (
    _PROJECT_ROOT / "build" / "core_engine",
    _PROJECT_ROOT / "build" / "core_engine" / "Release",
    _PROJECT_ROOT,
):
    _p = str(_candidate)
    if _candidate.exists() and _p not in sys.path:
        sys.path.insert(0, _p)

import ar_helmet_core as core_engine
ar_helmet_core = core_engine

from app_ui.one_euro_filter import OneEuroFilter

RENDER_MODE_LABELS = {
    0: "Modo: Realista",
    1: "Modo: Cel-Shading",
    2: "Modo: Cyberpunk",
}

# Códigos de teclas F1, F2, F3 en OpenCV waitKey / waitKeyEx (Linux X11, Windows Win32, Qt) y fallback '1', '2', '3'
KEY_TO_RENDER_MODE = {
    ord("1"): 0,
    ord("2"): 1,
    ord("3"): 2,
    65470: 0,       
    65471: 1,       
    65472: 2,       
    190: 0,         
    191: 1,         
    192: 2,         
    7340032: 0,     
    7405568: 1,     
    7471104: 2,     
    0x01000030: 0,  
    0x01000031: 1,  
    0x01000032: 2,  
}

try:
    from app_ui.gui import MainWindow
    from app_ui.landmark_detector import LandmarkDetector
    from app_ui.camera import CameraStream
except ImportError:
    from gui import MainWindow
    from landmark_detector import LandmarkDetector
    from camera import CameraStream


def get_opengl_proj_from_opencv(cam_mat, width, height, near, far):
    """Convierte la matriz intrínseca de OpenCV en una Matriz de Proyección de OpenGL."""
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


class VideoPipelineWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)

    def __init__(self, camera_index: int = 1):
        super().__init__()
        self._mutex = QMutex()
        self._running = True
        self._camera_index = camera_index
        self._pending_camera_index = camera_index
        self._mask_enabled = True
        self._render_mode = 0

        # Cola segura para cambio de modelo y parámetros de transformación en vivo
        self._pending_model_path = None
        self._transform_params = {
            "scale": 450.0,
            "offset": [0.0, -55.0, 180.0],
            "rot": [0.0, 0.0, 0.0],
        }

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

    def change_model(self, path: str):
        self._mutex.lock()
        self._pending_model_path = path
        self._mutex.unlock()

    def update_transform(self, params: dict):
        self._mutex.lock()
        self._transform_params = {
            "scale": float(params["scale"]),
            "offset": list(params["offset"]),
            "rot": list(params["rot"]),
        }
        self._mutex.unlock()

    def set_render_mode(self, mode: int):
        self._mutex.lock()
        self._render_mode = max(0, min(2, int(mode)))
        self._mutex.unlock()

    def handle_key(self, key: int) -> bool:
        """Mapea teclas F1/F2/F3 y fallback '1'/'2'/'3' (compatible con cv2.waitKey y eventos Qt)."""
        if key in KEY_TO_RENDER_MODE:
            self.set_render_mode(KEY_TO_RENDER_MODE[key])
            return True
        if key != -1 and (key & 0xFF) in KEY_TO_RENDER_MODE:
            self.set_render_mode(KEY_TO_RENDER_MODE[key & 0xFF])
            return True
        return False

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()
        self.wait()

    def run(self):
        print("[VideoWorker] Levantando motor C++ Offscreen...")
        light_estimator = ar_helmet_core.LightEstimator()
        motor = ar_helmet_core.Renderer(1280, 720)

        # Shaders del pipeline compuesto en GPU
        shader = ar_helmet_core.Shader("core_engine/shaders/basic.vert", "core_engine/shaders/basic.frag")
        bg_shader = ar_helmet_core.Shader("core_engine/shaders/camera_bg.vert", "core_engine/shaders/camera_bg.frag")

        modelo = None
        default_model = "assets/models/casco_v2.obj"
        if os.path.exists(default_model):
            modelo = ar_helmet_core.Model(default_model)

        # Filtros One-Euro para estabilizar rotación y traslación
        filter_rvec = OneEuroFilter(min_cutoff=1.2, beta=0.8, d_cutoff=1.0)
        filter_tvec = OneEuroFilter(min_cutoff=0.8, beta=0.03, d_cutoff=1.0)

        last_rvec = None
        last_tvec = None
        lost_frames = 0
        MAX_LOST_FRAMES = 4

        stream = CameraStream(self._camera_index).start()
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
                stream.change_camera(self._camera_index)

            # Cambio de modelo 3D en caliente (dentro del contexto OpenGL activo)
            new_model_path = self._pending_model_path
            self._pending_model_path = None

            mask_on = self._mask_enabled
            render_mode = self._render_mode
            t_params = {
                "scale": self._transform_params["scale"],
                "offset": list(self._transform_params["offset"]),
                "rot": list(self._transform_params["rot"]),
            }
            self._mutex.unlock()

            if new_model_path is not None and os.path.exists(new_model_path):
                try:
                    print(f"[VideoWorker] Cargando modelo en GPU: {new_model_path}")
                    modelo = ar_helmet_core.Model(new_model_path)
                except Exception as e:
                    print(f"[VideoWorker] Error al cargar {new_model_path}: {e}")

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

            ret, frame = stream.read()
            if not ret or frame is None:
                self.msleep(2)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Inicialización de matriz intrínseca de cámara
            if self._cam_matrix is None:
                self._cam_matrix = np.array([
                    [w, 0, w / 2.0],
                    [0, w, h / 2.0],
                    [0, 0, 1.0]
                ], dtype=np.float64)

            # Estimación de pose y renderizado AR compuesto
            if mask_on and modelo is not None:
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
                    # Persistencia de pose durante pérdida transitoria
                    lost_frames += 1
                    rvec = last_rvec
                    tvec = last_tvec
                else:
                    # Pérdida definitiva: reinicio de filtros y solver
                    lost_frames += 1
                    last_rvec = None
                    last_tvec = None
                    filter_rvec.reset()
                    filter_tvec.reset()
                    self._detector.reset_pose()
                    rvec, tvec = None, None

                # Renderizado compuesto en GPU si la pose es válida
                if rvec is not None and tvec is not None:
                    rmat, _ = cv2.Rodrigues(rvec)

                    view_mat = np.eye(4, dtype=np.float32)
                    view_mat[:3, :3] = rmat
                    view_mat[:3, 3] = tvec.flatten()

                    # Adaptar sistema de coordenadas OpenCV a OpenGL
                    cv_to_gl = np.array([
                        [ 1,  0,  0,  0],
                        [ 0, -1,  0,  0],
                        [ 0,  0, -1,  0],
                        [ 0,  0,  0,  1]
                    ], dtype=np.float32)
                    view_mat = cv_to_gl @ view_mat

                    proj_mat = get_opengl_proj_from_opencv(self._cam_matrix, w, h, 0.1, 10000.0)

                    # Construcción dinámica de Model Matrix (Escala + Rotación local + Offset XYZ)
                    escala_real = t_params["scale"]
                    offset_x, offset_y, offset_z = t_params["offset"]
                    rot_x, rot_y, rot_z = t_params["rot"]

                    scale_mat = np.diag([escala_real, -escala_real, escala_real, 1.0]).astype(np.float32)

                    rx, ry, rz = np.deg2rad(rot_x), np.deg2rad(rot_y), np.deg2rad(rot_z)
                    rot_local, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float32))
                    rot_mat = np.eye(4, dtype=np.float32)
                    rot_mat[:3, :3] = rot_local

                    model_mat = scale_mat @ rot_mat
                    model_mat[0, 3] = offset_x
                    model_mat[1, 3] = offset_y
                    model_mat[2, 3] = offset_z

                    light_dir, intensity, _ = light_estimator.estimate(frame)

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

                    core_engine.set_render_mode(render_mode)
                    motor.set_render_mode(render_mode)
                    motor.update_camera_frame(frame)
                    motor.render_scene(modelo, shader, bg_shader)
                    frame = cv2.flip(motor.get_frame(), 0)

            # HUD en el frame indicando el modo de sombreado activo
            hud_text = RENDER_MODE_LABELS.get(render_mode, "Modo: Realista")
            cv2.putText(
                frame,
                hud_text,
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 229, 255),
                2,
                cv2.LINE_AA,
            )

            # Cálculo y suavizado de FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            self.frame_ready.emit(frame)
            self.fps_updated.emit(fps)

        stream.stop()
        del shader
        del motor


def main():
    app = QApplication(sys.argv)
    window = MainWindow()

    worker = VideoPipelineWorker(camera_index=1)

    # Conexión de señales y slots desacoplados
    worker.frame_ready.connect(window.update_frame)
    worker.fps_updated.connect(window.update_fps)
    window.mask_toggled.connect(worker.set_mask_enabled)
    window.camera_changed.connect(worker.change_camera)
    window.model_changed.connect(worker.change_model)
    window.transform_changed.connect(worker.update_transform)
    window.render_mode_changed.connect(worker.set_render_mode)
    window.window_closed.connect(worker.stop)

    # Sincronizar el modelo seleccionado al iniciar
    window.sync_initial_state()

    worker.start()
    window.show()

    exit_code = app.exec()
    worker.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()