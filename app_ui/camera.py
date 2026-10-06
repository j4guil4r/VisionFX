import cv2
import time
import numpy as np
from landmark_detector import LandmarkDetector

# Ejes 3D de prueba (X: Derecha, Y: Arriba, Z: Frontal)
AXES_3D = np.array([
    [0.0, 0.0, 0.0],
    [60.0, 0.0, 0.0],
    [0.0, -60.0, 0.0],
    [0.0, 0.0, -60.0]
], dtype=np.float64)

def main():
    cap = cv2.VideoCapture(0)
    w_cam, h_cam = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_cam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_cam)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara.")
        return

    # Construir la matriz intrínseca de la cámara
    cam_matrix = np.array([[w_cam, 0, w_cam / 2.0],
                           [0, w_cam, h_cam / 2.0],
                           [0, 0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    detector = LandmarkDetector()
    print("[INFO] Tracking 3D iniciado. Presiona 'Q' para salir.")
    
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Extraer puntos 2D y calcular pose 3D
        points_2d = detector.get_landmarks(frame)
        if points_2d:
            pose = detector.estimate_pose(points_2d, cam_matrix, dist_coeffs)
            
            if pose:
                rvec, tvec = pose
                
                # Proyectar ejes ortogonales al frame 2D
                proj_axes, _ = cv2.projectPoints(AXES_3D, rvec, tvec, cam_matrix, dist_coeffs)
                pts = proj_axes.reshape(-1, 2).astype(int)
                origin = tuple(pts[0])
                
                # Dibujar X (Rojo), Y (Verde), Z (Azul)
                cv2.line(frame, origin, tuple(pts[1]), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(frame, origin, tuple(pts[2]), (0, 255, 0), 3, cv2.LINE_AA)
                cv2.line(frame, origin, tuple(pts[3]), (255, 0, 0), 3, cv2.LINE_AA)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("AR Helmet - Tracking 3D", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()