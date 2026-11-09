import json
import os
from pathlib import Path
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

CALIBRATION_FILE = Path("assets/models/calibration.json")

DEFAULT_CALIBRATION = {
    "scale": 450.0,
    "offset": [0.0, -55.0, 180.0],  # [X, Y, Z]
    "rot": [0.0, 0.0, 0.0],         # [Pitch_X, Yaw_Y, Roll_Z] en grados
}


class MainWindow(QMainWindow):
    # Señales desacopladas para interactuar de forma Thread-Safe con el worker
    mask_toggled = pyqtSignal(bool)
    camera_changed = pyqtSignal(int)
    model_changed = pyqtSignal(str)
    transform_changed = pyqtSignal(dict)
    render_mode_changed = pyqtSignal(int)
    window_closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionFX - AR Pipeline UI")
        self.resize(1050, 740)
        self.setMinimumSize(850, 620)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Estado de calibración y persistencia
        self._calibrations = self._load_calibration_file()
        self._current_model_key = ""
        self._current_params = self._clone_params(DEFAULT_CALIBRATION)

        # Máquina de estados estilo Blender
        self._modal_mode = None      # None, 'T' (Traslación), 'R' (Rotación), 'S' (Escala)
        self._modal_axis = None      # None, 0 ('X'), 1 ('Y'), 2 ('Z')
        self._input_buffer = ""      # Buffer para escribir números exactos (ej. "-55.0")
        self._backup_params = None   # Copia para revertir si se presiona ESC

        self._init_ui()
        self._apply_style()
        self._populate_models()

    # -------------------------------------------------------------------------
    # Persistencia JSON por modelo
    # -------------------------------------------------------------------------
    def _clone_params(self, data: dict) -> dict:
        return {
            "scale": float(data.get("scale", 450.0)),
            "offset": [float(x) for x in data.get("offset", [0.0, -55.0, 180.0])],
            "rot": [float(x) for x in data.get("rot", [0.0, 0.0, 0.0])],
        }

    def _load_calibration_file(self) -> dict:
        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[UI] Advertencia: No se pudo leer {CALIBRATION_FILE}: {e}")
        return {}

    def _save_calibration_file(self):
        if not self._current_model_key:
            return
        self._calibrations[self._current_model_key] = self._clone_params(self._current_params)
        try:
            CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump(self._calibrations, f, indent=4)
            print(f"[UI] Calibración guardada para '{self._current_model_key}'")
        except Exception as e:
            print(f"[UI] Error guardando calibración: {e}")

    # -------------------------------------------------------------------------
    # Construcción de Interfaz
    # -------------------------------------------------------------------------
    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        # Barra superior HUD (Comandos estilo Blender y estado actual)
        self.hud_label = QLabel()
        self.hud_label.setObjectName("hudLabel")
        self.hud_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        main_layout.addWidget(self.hud_label)

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
        controls_layout.setSpacing(16)

        # Control: Alternar máscara
        self.mask_checkbox = QCheckBox("Máscara 3D")
        self.mask_checkbox.setChecked(True)
        self.mask_checkbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mask_checkbox.toggled.connect(self.mask_toggled.emit)
        controls_layout.addWidget(self.mask_checkbox)

        # Control: Selector dinámico de Modelos .OBJ
        model_label = QLabel("Modelo 3D:")
        controls_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        controls_layout.addWidget(self.model_combo)

        # Control: Selector de cámara
        cam_label = QLabel("Cámara:")
        controls_layout.addWidget(cam_label)

        self.camera_combo = QComboBox()
        self.camera_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.camera_combo.addItems(["Cámara 0", "Cámara 1", "Cámara 2"])
        self.camera_combo.currentIndexChanged.connect(self.camera_changed.emit)
        controls_layout.addWidget(self.camera_combo)

        controls_layout.addStretch()

        # Display: FPS
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setObjectName("fpsLabel")
        controls_layout.addWidget(self.fps_label)

        main_layout.addWidget(controls_frame)
        self._update_hud_text()

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1e;
            }
            #controlsFrame {
                background-color: #24242a;
                border-radius: 8px;
            }
            #hudLabel {
                background-color: #202026;
                color: #00e5ff;
                border: 1px solid #32323c;
                border-radius: 6px;
                padding: 8px 12px;
                font-family: 'Consolas', 'Monospace', monospace;
                font-size: 13px;
                font-weight: bold;
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

    # -------------------------------------------------------------------------
    # Gestión de Modelos (.obj)
    # -------------------------------------------------------------------------
    def _populate_models(self):
        models_dir = Path("assets/models")
        models_dir.mkdir(parents=True, exist_ok=True)

        obj_files = sorted(models_dir.rglob("*.obj"))
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        for obj_path in obj_files:
            rel_name = obj_path.relative_to(models_dir).as_posix()
            self.model_combo.addItem(rel_name, str(obj_path))

        self.model_combo.blockSignals(False)

        if self.model_combo.count() > 0:
            # Preferir casco_v22.obj si existe, si no el primero de la lista
            default_idx = 0
            for i in range(self.model_combo.count()):
                if "casco_v22.obj" in self.model_combo.itemText(i):
                    default_idx = i
                    break
            self.model_combo.setCurrentIndex(default_idx)
            self._on_model_combo_changed(default_idx)

    def sync_initial_state(self):
        """Emite el modelo y transformación inicial cuando el worker ya está conectado."""
        if self.model_combo.count() > 0:
            self._on_model_combo_changed(self.model_combo.currentIndex())

    def _on_model_combo_changed(self, index: int):
        if index < 0:
            return
        full_path = self.model_combo.itemData(index)
        if not full_path or not os.path.exists(full_path):
            return

        # Cancelar cualquier edición modal previa
        self._modal_mode = None
        self._modal_axis = None
        self._input_buffer = ""

        # Cargar calibración guardada para este modelo (o usar default)
        self._current_model_key = os.path.basename(full_path)
        saved = self._calibrations.get(self._current_model_key, DEFAULT_CALIBRATION)
        self._current_params = self._clone_params(saved)

        self.model_changed.emit(full_path)
        self.transform_changed.emit(self._clone_params(self._current_params))
        self._update_hud_text()

    # -------------------------------------------------------------------------
    # Máquina de Estados Modal (Atajos estilo Blender: T/G, R, S + Ejes X, Y, Z)
    # -------------------------------------------------------------------------
    def _update_hud_text(self):
        ox, oy, oz = self._current_params["offset"]
        rx, ry, rz = self._current_params["rot"]
        sc = self._current_params["scale"]

        if self._modal_mode is None:
            self.hud_label.setStyleSheet("background-color: #202026; color: #00e5ff; border: 1px solid #32323c; border-radius: 6px; padding: 8px 12px; font-family: monospace; font-weight: bold;")
            self.hud_label.setText(
                f"[ATAJOS]  F1/1: Realista | F2/2: Cel-Shading | F3/3: Cyberpunk  ||  T: Mover | R: Rotar | S: Escala   ||   "
                f"Pos({ox:.0f}, {oy:.0f}, {oz:.0f}) | Rot({rx:.0f}°, {ry:.0f}°, {rz:.0f}°) | Escala: {sc:.0f}"
            )
            return

        # Estilo activo (naranja Blender)
        self.hud_label.setStyleSheet("background-color: #332414; color: #ffab40; border: 1px solid #ff9100; border-radius: 6px; padding: 8px 12px; font-family: monospace; font-weight: bold;")

        mode_names = {"T": "TRASLACIÓN", "R": "ROTACIÓN", "S": "ESCALA"}
        axis_names = {0: "X", 1: "Y", 2: "Z"}
        m_name = mode_names[self._modal_mode]

        if self._modal_mode in ("T", "R") and self._modal_axis is None:
            self.hud_label.setText(
                f"[{m_name}] Presiona el eje a modificar: [X] , [Y] o [Z]   (ESC para cancelar)"
            )
            return

        if self._modal_mode == "S":
            val_str = self._input_buffer if self._input_buffer else f"{sc:.1f}"
            self.hud_label.setText(
                f"[ESCALA] Valor: {val_str}  |  Escribe número o usa ↑/↓ (Shift=fino)  |  ENTER: Guardar | ESC: Cancelar"
            )
        else:
            ax_char = axis_names[self._modal_axis]
            curr_val = (
                self._current_params["offset"][self._modal_axis]
                if self._modal_mode == "T"
                else self._current_params["rot"][self._modal_axis]
            )
            val_str = self._input_buffer if self._input_buffer else f"{curr_val:.1f}"
            unit = "mm" if self._modal_mode == "T" else "°"
            self.hud_label.setText(
                f"[{m_name} EJE {ax_char}] Valor: {val_str}{unit}  |  Escribe número o usa ↑/↓ (Shift=fino)  |  ENTER: Guardar | ESC: Cancelar"
            )

    def _apply_typed_buffer(self):
        """Convierte el texto tipeado en valor numérico en vivo si es válido."""
        if not self._input_buffer or self._input_buffer in ("-", ".", "-."):
            return
        try:
            val = float(self._input_buffer)
        except ValueError:
            return

        if self._modal_mode == "S":
            self._current_params["scale"] = max(10.0, val)
        elif self._modal_mode == "T" and self._modal_axis is not None:
            self._current_params["offset"][self._modal_axis] = val
        elif self._modal_mode == "R" and self._modal_axis is not None:
            self._current_params["rot"][self._modal_axis] = val

        self.transform_changed.emit(self._clone_params(self._current_params))

    def _step_current_value(self, direction: float, fine: bool):
        """Incrementa o disminuye el valor con las flechas del teclado."""
        self._input_buffer = ""  # Limpiar buffer de texto al usar flechas
        step = 1.0 if fine else 5.0

        if self._modal_mode == "S":
            self._current_params["scale"] = max(10.0, self._current_params["scale"] + direction * step * 2.0)
        elif self._modal_mode == "T" and self._modal_axis is not None:
            self._current_params["offset"][self._modal_axis] += direction * step
        elif self._modal_mode == "R" and self._modal_axis is not None:
            self._current_params["rot"][self._modal_axis] = (
                self._current_params["rot"][self._modal_axis] + direction * step
            ) % 360.0

        self.transform_changed.emit(self._clone_params(self._current_params))
        self._update_hud_text()

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        # Hot-switching de Shaders: F1, F2, F3 (y fallback numérico 1, 2, 3 fuera de edición modal)
        if key == Qt.Key.Key_F1 or (self._modal_mode is None and key == Qt.Key.Key_1):
            self.render_mode_changed.emit(0)
            return
        if key == Qt.Key.Key_F2 or (self._modal_mode is None and key == Qt.Key.Key_2):
            self.render_mode_changed.emit(1)
            return
        if key == Qt.Key.Key_F3 or (self._modal_mode is None and key == Qt.Key.Key_3):
            self.render_mode_changed.emit(2)
            return

        # Atajo Ctrl+R: Resetear modelo actual a valores por defecto
        if key == Qt.Key.Key_R and (mods & Qt.KeyboardModifier.ControlModifier):
            self._current_params = self._clone_params(DEFAULT_CALIBRATION)
            self._modal_mode = None
            self._modal_axis = None
            self._input_buffer = ""
            self._save_calibration_file()
            self.transform_changed.emit(self._clone_params(self._current_params))
            self._update_hud_text()
            return

        # 1. Si estamos dentro de un modo interactivo: manejar ESC y ENTER
        if self._modal_mode is not None:
            if key == Qt.Key.Key_Escape:
                # Cancelar y restaurar valores previos
                if self._backup_params is not None:
                    self._current_params = self._clone_params(self._backup_params)
                    self.transform_changed.emit(self._clone_params(self._current_params))
                self._modal_mode = None
                self._modal_axis = None
                self._input_buffer = ""
                self._update_hud_text()
                return

            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # Confirmar y guardar en JSON
                self._save_calibration_file()
                self._modal_mode = None
                self._modal_axis = None
                self._input_buffer = ""
                self._update_hud_text()
                return

        # 2. Selección de Modo Principal: T (o G) = Trasladar, R = Rotar, S = Escalar
        if key in (Qt.Key.Key_T, Qt.Key.Key_G):
            if self._modal_mode is None:
                self._backup_params = self._clone_params(self._current_params)
            self._modal_mode = "T"
            self._modal_axis = None
            self._input_buffer = ""
            self._update_hud_text()
            return

        if key == Qt.Key.Key_R:
            if self._modal_mode is None:
                self._backup_params = self._clone_params(self._current_params)
            self._modal_mode = "R"
            self._modal_axis = None
            self._input_buffer = ""
            self._update_hud_text()
            return

        if key == Qt.Key.Key_S:
            if self._modal_mode is None:
                self._backup_params = self._clone_params(self._current_params)
            self._modal_mode = "S"
            self._modal_axis = None
            self._input_buffer = ""
            self._update_hud_text()
            return

        # 3. Selección de Eje (X, Y, Z) cuando estamos en Traslación o Rotación
        if self._modal_mode in ("T", "R"):
            if key == Qt.Key.Key_X:
                self._modal_axis = 0
                self._input_buffer = ""
                self._update_hud_text()
                return
            if key == Qt.Key.Key_Y:
                self._modal_axis = 1
                self._input_buffer = ""
                self._update_hud_text()
                return
            if key == Qt.Key.Key_Z:
                self._modal_axis = 2
                self._input_buffer = ""
                self._update_hud_text()
                return

        # 4. Ajustes de valor (cuando el modo S está activo, o T/R ya tienen eje elegido)
        can_edit_value = (self._modal_mode == "S") or (
            self._modal_mode in ("T", "R") and self._modal_axis is not None
        )

        if can_edit_value:
            fine_step = bool(mods & Qt.KeyboardModifier.ShiftModifier)

            if key in (Qt.Key.Key_Up, Qt.Key.Key_Right, Qt.Key.Key_Plus):
                self._step_current_value(+1.0, fine_step)
                return
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Minus):
                # Si el buffer está vacío y presiona '-', iniciar número negativo
                if key == Qt.Key.Key_Minus and self._input_buffer == "":
                    self._input_buffer = "-"
                    self._update_hud_text()
                    return
                self._step_current_value(-1.0, fine_step)
                return

            if key == Qt.Key.Key_Backspace:
                self._input_buffer = self._input_buffer[:-1]
                self._apply_typed_buffer()
                self._update_hud_text()
                return

            # Captura de dígitos y punto decimal
            text = event.text()
            if text and (text.isdigit() or text in (".", "-")):
                if text == "." and "." in self._input_buffer:
                    return
                if text == "-" and self._input_buffer != "":
                    return
                self._input_buffer += text
                self._apply_typed_buffer()
                self._update_hud_text()
                return

        super().keyPressEvent(event)

    # -------------------------------------------------------------------------
    # Drag & Drop y Actualización de Video
    # -------------------------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].toLocalFile().lower().endswith(".obj"):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if not os.path.exists(file_path):
            return
        # Buscar si ya está en el combo, si no, agregarlo
        for i in range(self.model_combo.count()):
            if os.path.abspath(self.model_combo.itemData(i)) == os.path.abspath(file_path):
                self.model_combo.setCurrentIndex(i)
                return
        self.model_combo.addItem(os.path.basename(file_path), file_path)
        self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def update_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)

        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def closeEvent(self, event):
        self.window_closed.emit()
        super().closeEvent(event)