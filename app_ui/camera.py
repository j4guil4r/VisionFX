import cv2
import time
from landmark_detector import LandmarkDetector

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara web.")
        return
    
    detector = LandmarkDetector()
    print("[INFO] Cámara y detector MediaPipe iniciados. Presiona 'Q' para salir.")
    
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Extraemos los 6 puntos de interés de la cara
        points_2d = detector.get_landmarks(frame)
        
        # Si encuentra un rostro, dibujamos los puntos para verificar visualmente
        if points_2d:
            for pt in points_2d:
                cv2.circle(frame, pt, 4, (0, 255, 255), -1)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("AR Helmet - Captura Base", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()