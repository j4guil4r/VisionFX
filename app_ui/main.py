import sys
import time
import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

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
                        proj_axes, _ = cv2.projectPoints(
                            AXES_3D, rvec, tvec, self._cam_matrix, self._dist_coeffs
                        )
                        pts = proj_axes.reshape(-1, 2).astype(int)
                        origin = tuple(pts[0])
                        cv2.line(frame, origin, tuple(pts[1]), (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.line(frame, origin, tuple(pts[2]), (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.line(frame, origin, tuple(pts[3]), (255, 0, 0), 3, cv2.LINE_AA)

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
