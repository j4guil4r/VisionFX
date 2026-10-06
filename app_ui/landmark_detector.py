import cv2
import mediapipe as mp

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
        
        # Índices antropométricos canónicos para PnP (Farkas / ISO)
        # 1: Punta nariz, 199: Mentón, 33/263: Ojos ext, 61/291: Boca
        self.landmark_indices = [1, 199, 33, 263, 61, 291]

    def get_landmarks(self, frame_bgr):
        """Procesa el frame y devuelve las coordenadas (x,y) de los 6 puntos clave."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        
        points_2d = []
        for idx in self.landmark_indices:
            lm = face.landmark[idx]
            # Convertir coordenadas normalizadas (0.0 - 1.0) a píxeles reales
            x, y = int(lm.x * w), int(lm.y * h)
            points_2d.append((x, y))
            
        return points_2d