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
            points_2d.append((int(lm.x * w), int(lm.y * h)))
        return points_2d

    def estimate_pose(self, points_2d, cam_matrix, dist_coeffs):
        """Calcula la orientación (rvec) y posición (tvec) de la cabeza."""
        if not points_2d:
            return None
            
        pts_2d_np = np.array(points_2d, dtype=np.float64)
        
        # Resolución iterativa del problema PnP
        success, rvec, tvec = cv2.solvePnP(
            self.FACE_3D, pts_2d_np, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            return rvec, tvec
        return None