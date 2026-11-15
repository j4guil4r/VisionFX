import os
import sys
import time
import cv2
from pathlib import Path
import numpy as np
from PyQt6.QtCore import QMutex, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication
import json


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



class TouchlessZone:
    """Zona interactiva AR en pantalla activada al pasar la mano (diferencia óptica de frames)."""
    BASE_W = 1280.0
    BASE_H = 720.0

    def __init__(self, rect: tuple[float, float, float, float], label: str, color: tuple[int, int, int]):
        x1, y1, x2, y2 = rect
        if max(x1, y1, x2, y2) > 1.0:
            self.rel_rect = (
                float(x1) / self.BASE_W,
                float(y1) / self.BASE_H,
                float(x2) / self.BASE_W,
                float(y2) / self.BASE_H,
            )
        else:
            self.rel_rect = (float(x1), float(y1), float(x2), float(y2))
        self.label = label
        self.color = color
        self.prev_gray = None
        self.last_trigger = 0.0
        self.cooldown = 1.2
        self.flash_until = 0.0

    def _get_pixel_rect(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        w, h = frame.shape[1], frame.shape[0]
        rx1, ry1, rx2, ry2 = self.rel_rect
        x1 = max(0, min(w - 1, int(round(rx1 * w))))
        y1 = max(0, min(h - 1, int(round(ry1 * h))))
        x2 = max(x1 + 1, min(w, int(round(rx2 * w))))
        y2 = max(y1 + 1, min(h, int(round(ry2 * h))))
        return x1, y1, x2, y2

    def check_motion(self, frame: np.ndarray) -> bool:
        if frame is None or frame.size == 0:
            return False
        x1, y1, x2, y2 = self._get_pixel_rect(frame)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        if self.prev_gray is not None and self.prev_gray.shape != gray.shape:
            self.prev_gray = None

        triggered = False
        now = time.time()
        if self.prev_gray is not None and (now - self.last_trigger) > self.cooldown:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_ratio = float(np.count_nonzero(thresh)) / float(max(1, thresh.size))
            if motion_ratio > 0.22:
                self.last_trigger = now
                self.flash_until = now + 0.35
                triggered = True

        self.prev_gray = gray
        return triggered

    def draw(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return
        w, h = frame.shape[1], frame.shape[0]
        x1, y1, x2, y2 = self._get_pixel_rect(frame)
        now = time.time()
        active_color = (0, 255, 120) if now < self.flash_until else self.color

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), active_color, -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), active_color, 2)
        font_scale = max(0.38, min(0.65, 0.55 * (w / self.BASE_W)))
        cv2.putText(
            frame,
            self.label,
            (x1 + int(12 * (w / self.BASE_W)), (y1 + y2) // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _draw_ocular_surge_fx(frame: np.ndarray, eye_centers: list[tuple[int, int]]):
    """Dibuja núcleos de plasma y rayos eléctricos procedurales en los ojos durante la sobrecarga."""
    if not eye_centers or frame is None or frame.size == 0:
        return
    h, w = frame.shape[:2]
    rng = np.random.default_rng(int(time.time() * 30.0))
    overlay = frame.copy()

    for ex, ey in eye_centers:
        if not (0 <= ex < w and 0 <= ey < h):
            continue
        # Halo de plasma exterior e interior
        cv2.circle(overlay, (ex, ey), 22, (255, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (ex, ey), 8, (255, 255, 255), -1, cv2.LINE_AA)

        # 4 arcos eléctricos quebrados irradiando desde cada ojo
        for _ in range(4):
            pts = [(ex, ey)]
            cx, cy = float(ex), float(ey)
            angle = rng.uniform(0, 2.0 * np.pi)
            for _step in range(4):
                length = rng.uniform(12.0, 26.0)
                angle += rng.uniform(-0.65, 0.65)
                cx += np.cos(angle) * length
                cy += np.sin(angle) * length
                pts.append((int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))))
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i + 1], (255, 230, 0), 4, cv2.LINE_AA)
                cv2.line(frame, pts[i], pts[i + 1], (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)


class VideoPipelineWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    model_synced = pyqtSignal(str)

    def __init__(self, camera_index: int = 0):
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
        self._cam_dims = (0, 0)
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        models_dir = _PROJECT_ROOT / "assets" / "models"
        self._available_models = sorted(str(p.resolve()) for p in models_dir.rglob("*.obj"))
        self._current_model_idx = 0

    def _apply_saved_calibration(self, model_path: str):
        cal_file = _PROJECT_ROOT / "assets" / "models" / "calibration.json"
        default_cfg = {"scale": 450.0, "offset": [0.0, -55.0, 180.0], "rot": [0.0, 0.0, 0.0]}
        cfg = default_cfg
        if cal_file.exists():
            try:
                data = json.loads(cal_file.read_text(encoding="utf-8"))
                cfg = data.get(Path(model_path).name, default_cfg)
            except Exception:
                cfg = default_cfg
        self._transform_params = {
            "scale": float(cfg.get("scale", 450.0)),
            "offset": list(cfg.get("offset", [0.0, -55.0, 180.0])),
            "rot": list(cfg.get("rot", [0.0, 0.0, 0.0])),
        }

    def set_mask_enabled(self, enabled: bool):
        self._mutex.lock()
        try:
            self._mask_enabled = enabled
        finally:
            self._mutex.unlock()

    def change_camera(self, index: int):
        self._mutex.lock()
        try:
            self._pending_camera_index = index
        finally:
            self._mutex.unlock()

    def change_model(self, path: str):
        resolved = str(Path(path).resolve()) if path else path
        self._mutex.lock()
        try:
            self._pending_model_path = resolved
            if resolved in self._available_models:
                self._current_model_idx = self._available_models.index(resolved)
        finally:
            self._mutex.unlock()

    def update_transform(self, params: dict):
        self._mutex.lock()
        try:
            self._transform_params = {
                "scale": float(params["scale"]),
                "offset": list(params["offset"]),
                "rot": list(params["rot"]),
            }
        finally:
            self._mutex.unlock()

    def set_render_mode(self, mode: int):
        self._mutex.lock()
        try:
            self._render_mode = max(0, min(2, int(mode)))
        finally:
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
        try:
            self._running = False
        finally:
            self._mutex.unlock()
        self.wait()

    def run(self):
        print("[VideoWorker] Levantando motor C++ Offscreen...")
        light_estimator = ar_helmet_core.LightEstimator()
        motor = ar_helmet_core.Renderer(1280, 720)

        # Shaders del pipeline compuesto en GPU
        shader = ar_helmet_core.Shader(
            str(_PROJECT_ROOT / "core_engine" / "shaders" / "basic.vert"),
            str(_PROJECT_ROOT / "core_engine" / "shaders" / "basic.frag"),
        )
        bg_shader = ar_helmet_core.Shader(
            str(_PROJECT_ROOT / "core_engine" / "shaders" / "camera_bg.vert"),
            str(_PROJECT_ROOT / "core_engine" / "shaders" / "camera_bg.frag"),
        )

        modelo = None
        default_model = str(_PROJECT_ROOT / "assets" / "models" / "casco_v2.obj")
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

        zone_model = TouchlessZone((30, 85, 260, 165), "MANO: CASCO", (0, 229, 255))
        zone_shader = TouchlessZone((1020, 85, 1250, 165), "MANO: SHADER", (255, 180, 0))

        while True:
            self._mutex.lock()
            try:
                if not self._running:
                    break

                # Cambio de cámara en caliente (extraemos índice para no bloquear el mutex durante I/O)
                cam_to_switch = None
                if self._pending_camera_index != self._camera_index:
                    self._camera_index = self._pending_camera_index
                    cam_to_switch = self._camera_index

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
            finally:
                self._mutex.unlock()

            if cam_to_switch is not None:
                stream.change_camera(cam_to_switch)
                self._cam_matrix = None

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

            if zone_model.check_motion(frame) and self._available_models:
                self._current_model_idx = (self._current_model_idx + 1) % len(self._available_models)
                next_model = self._available_models[self._current_model_idx]
                try:
                    print(f"[VideoWorker] Cambio gestual de modelo en GPU: {next_model}")
                    modelo = ar_helmet_core.Model(next_model)
                    self._mutex.lock()
                    try:
                        self._apply_saved_calibration(next_model)
                        t_params = {
                            "scale": self._transform_params["scale"],
                            "offset": list(self._transform_params["offset"]),
                            "rot": list(self._transform_params["rot"]),
                        }
                    finally:
                        self._mutex.unlock()
                    self.model_synced.emit(next_model)
                except Exception as e:
                    print(f"[VideoWorker] Error en cambio gestual al cargar {next_model}: {e}")

            if zone_shader.check_motion(frame):
                render_mode = (render_mode + 1) % 3
                self.set_render_mode(render_mode)

            w, h = frame.shape[1], frame.shape[0]

            # Inicialización o actualización de matriz intrínseca de cámara según resolución real
            if self._cam_matrix is None or self._cam_dims != (w, h):
                self._cam_dims = (w, h)
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

                    mouth_surge = self._detector.mouth_open_ratio > 0.13
                    if mouth_surge:
                        pulse = 1.8 + 0.5 * np.sin(time.time() * 16.0)
                        shader.set_float("lightIntensity", float(intensity) * float(pulse))
                        shader.set_vec3("ambientColor", 0.15, 0.45, 0.65)
                        shader.set_float("u_Surge", 1.0)
                    else:
                        shader.set_float("lightIntensity", float(intensity))
                        shader.set_vec3("ambientColor", 0.15, 0.15, 0.15)
                        shader.set_float("u_Surge", 0.0)

                    motor.update_camera_frame(frame)
                    motor.render_scene(modelo, shader, bg_shader)
                    frame = cv2.flip(motor.get_frame(), 0)

                    #if mouth_surge and self._detector.eye_centers_2d:
                        #_draw_ocular_surge_fx(frame, self._detector.eye_centers_2d)

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

            zone_model.draw(frame)
            zone_shader.draw(frame)
            if mask_on and self._detector.mouth_open_ratio > 0.13:
                cv2.putText(
                    frame,
                    "SOBRECARGA DE ENERGIA [BOCA]",
                    (24, int(200 * (h / 720.0))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 180),
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

    worker = VideoPipelineWorker(camera_index=0)

    # Conexión de señales y slots desacoplados
    worker.frame_ready.connect(window.update_frame)
    worker.fps_updated.connect(window.update_fps)
    worker.model_synced.connect(window.sync_model_from_worker)
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