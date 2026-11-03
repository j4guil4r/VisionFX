import cv2
import mediapipe as mp
import numpy as np


class LandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_indices = [1, 199, 33, 263, 61, 291]

        # Modelo 3D facial canónico en mm
        self.FACE_3D = np.array([
            [0.0, 0.0, 0.0],          # 1:   Punta de la nariz (Origen)
            [0.0, 110.0, 65.0],       # 199: Mentón
            [-75.0, -60.0, 65.0],     # 33:  Comisura ojo izquierdo
            [75.0, -60.0, 65.0],      # 263: Comisura ojo derecho
            [-40.0, 50.0, 35.0],      # 61:  Comisura boca izquierda
            [40.0, 50.0, 35.0]        # 291: Comisura boca derecha
        ], dtype=np.float64)

        # Memoria temporal de la pose para solvePnP continuo
        self.prev_rvec = None
        self.prev_tvec = None

    def get_landmarks(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        face = results.multi_face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        
        points_2d = []
        for idx in self.landmark_indices:
            lm = face.landmark[idx]
            # Usar float preserva la precisión subpíxel de MediaPipe
            points_2d.append((float(lm.x * w), float(lm.y * h)))
        return points_2d

    def estimate_pose(self, points_2d, cam_matrix, dist_coeffs):
        """Calcula la orientación (rvec) y posición (tvec) con fallback automático."""
        if not points_2d:
            self.reset_pose()
            return None
            
        pts_2d_np = np.array(points_2d, dtype=np.float64)
        success = False
        rvec, tvec = None, None

        # --- INTENTO 1: Continuidad temporal si hay pose previa ---
        if self.prev_rvec is not None and self.prev_tvec is not None:
            success, rvec, tvec = cv2.solvePnP(
                self.FACE_3D,
                pts_2d_np,
                cam_matrix,
                dist_coeffs,
                rvec=self.prev_rvec.copy(),
                tvec=self.prev_tvec.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            # Validación de cordura física: la cabeza debe estar frente a la cámara (Z > 0)
            if success and tvec[2] <= 0:
                success = False

        # --- INTENTO 2: Fallback limpio desde cero si el intento 1 falló o divergió ---
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                self.FACE_3D,
                pts_2d_np,
                cam_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success and tvec[2] <= 0:
                success = False

        # Guardar o limpiar según el resultado
        if success:
            self.prev_rvec = rvec.copy()
            self.prev_tvec = tvec.copy()
            return rvec, tvec
        else:
            self.reset_pose()
            return None

    def reset_pose(self):
        """Limpia la memoria temporal del solver."""
        self.prev_rvec = None
        self.prev_tvec = None