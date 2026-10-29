import threading
import platform
import time
import cv2
import numpy as np


class CameraStream:
    """Capturador asíncrono en hilo dedicado para desacoplar el bloqueo de I/O de la cámara."""

    def __init__(self, camera_index: int = 0):
        self._camera_index = camera_index
        self._cap = None
        self._ret = False
        self._frame = None
        self._running = False
        self._lock = threading.Lock()
        self._thread = None
        self._open(self._camera_index)

    def _open(self, index: int):
        system_os = platform.system()
        if system_os == "Linux":
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        elif system_os == "Windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        while self._running:
            if self._cap is not None and self._cap.isOpened():
                ret, frame = self._cap.read()
                with self._lock:
                    self._ret = ret
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if not self._ret or self._frame is None:
                return False, None
            # Copia contigua para evitar condiciones de carrera con el hilo de render
            return True, self._frame.copy()

    def is_opened(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def change_camera(self, index: int):
        with self._lock:
            self._camera_index = index
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()
            self._open(index)

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()