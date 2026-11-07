import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
import ar_helmet_core


class MainWindow(QMainWindow):
    # Señales desacopladas para interactuar con el pipeline
    mask_toggled = pyqtSignal(bool)
    camera_changed = pyqtSignal(int)
    window_closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionFX - AR Pipeline UI")
        self.resize(1000, 700)
        self.setMinimumSize(800, 600)
        self._init_ui()
        self._apply_style()
        self.setAcceptDrops(True)

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Visor de video
        self.video_label = QLabel("Esperando señal de video...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #121214; color: #888890; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 360)
        main_layout.addWidget(self.video_label, stretch=1)

        # Barra inferior de controles
        controls_frame = QFrame()
        controls_frame.setObjectName("controlsFrame")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(20)

        # Control: Alternar máscara
        self.mask_checkbox = QCheckBox("Máscara 3D")
        self.mask_checkbox.setChecked(True)
        self.mask_checkbox.toggled.connect(self.mask_toggled.emit)
        controls_layout.addWidget(self.mask_checkbox)

        # Control: Selector de cámara
        cam_label = QLabel("Cámara:")
        controls_layout.addWidget(cam_label)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Cámara 0", "Cámara 1", "Cámara 2"])
        self.camera_combo.currentIndexChanged.connect(self.camera_changed.emit)
        controls_layout.addWidget(self.camera_combo)

        controls_layout.addStretch()

        # Display: FPS
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setObjectName("fpsLabel")
        controls_layout.addWidget(self.fps_label)

        main_layout.addWidget(controls_frame)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1e;
            }
            #controlsFrame {
                background-color: #24242a;
                border-radius: 8px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            #fpsLabel {
                color: #00e676;
                font-weight: bold;
                font-size: 14px;
            }
            QCheckBox {
                color: #ffffff;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #555560;
                background-color: #2e2e38;
            }
            QCheckBox::indicator:checked {
                background-color: #00adb5;
                border-color: #00adb5;
            }
            QComboBox {
                background-color: #2e2e38;
                color: #ffffff;
                border: 1px solid #444450;
                border-radius: 4px;
                padding: 4px 10px;
                min-width: 110px;
            }
            QComboBox QAbstractItemView {
                background-color: #24242a;
                color: #ffffff;
                selection-background-color: #00adb5;
            }
        """)

    def update_frame(self, frame_bgr: np.ndarray):
        """Actualiza el visor con un frame en formato BGR de OpenCV."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        # Conversión directa BGR -> RGB a través del formato QImage
        qt_img = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)

        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_fps(self, fps: float):
        """Actualiza el display de FPS."""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, event):
        self.window_closed.emit()
        super().closeEvent(event)

    def dragEnterEvent(self, event):
        """Valida que el archivo arrastrado termine en .obj"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].toLocalFile().lower().endswith('.obj'):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        """Carga el nuevo modelo normalizado en tiempo real"""
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.load_new_model(file_path)

    def keyPressEvent(self, event):
        """Atajos de teclado rápidos para la galería durante la demo"""
        if event.key() == Qt.Key.Key_1:
            self.load_new_model("assets/models/helmet.obj")
        elif event.key() == Qt.Key.Key_2:
            self.load_new_model("assets/models/glasses.obj")
        elif event.key() == Qt.Key.Key_3:
            self.load_new_model("assets/models/mask.obj")

    def load_new_model(self, path: str):
        if not os.path.exists(path):
            return
        try:
            # Crea la nueva instancia de Model (se ejecuta el Auto-AABB en C++)
            nuevo_modelo = ar_helmet_core.Model(path)
            # Asigna la referencia atómicamente al hilo de renderizado
            self.pipeline_worker.modelo = nuevo_modelo
            print(f"[UI] Modelo cambiado exitosamente a: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error al cargar modelo", f"No se pudo cargar el archivo OBJ:\n{str(e)}")
