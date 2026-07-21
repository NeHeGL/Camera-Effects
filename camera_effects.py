# 
# Camera Effects - Native Desktop App
# 2026 Jeff Molofee (NeHe)
# Python + OpenCV + PyQt6 + MediaPipe (no browser needed)

import sys, os, time, datetime

# Suppress MediaPipe telemetry/clearcut logging
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
import cv2
import numpy as np
import mediapipe as mp


def _resource_path(relative):
    """Return absolute path to a bundled resource.

    When running as a PyInstaller frozen exe the data files live inside
    sys._MEIPASS (the temporary extraction folder).  When running from
    source they live next to this script.
    """
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QSlider, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QScrollArea, QSizePolicy, QFileDialog, QSplitter,
    QButtonGroup, QAbstractButton,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QIcon

from effects_lib import EFFECTS

# ── Camera capture thread ─────────────────────────────────────────
class CaptureThread(QThread):
    frame_ready = pyqtSignal(object)   # emits numpy BGR frame

    def __init__(self, device_id=0):
        super().__init__()
        self.device_id = device_id
        self._running  = False
        # True while the consumer (main-thread _on_frame) is still working
        # through the last emitted frame. Qt queues cross-thread emits, so
        # without this check a slow effect (background compositing + fx +
        # display taking longer than the ~33ms capture interval) causes
        # frames to queue up faster than they drain — a growing backlog
        # that eats memory and makes the feed feel increasingly laggy the
        # longer it runs. Skipping the emit instead always shows the
        # freshest frame and never lets a backlog build.
        self.frame_pending = False

    def run(self):
        cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._running = True
        while self._running:
            ret, frame = cap.read()
            if ret and not self.frame_pending:
                self.frame_pending = True
                self.frame_ready.emit(frame)
        cap.release()

    def stop(self):
        self._running = False
        self.wait(2000)

# ── Segmentation using MediaPipe ImageSegmenter (Tasks API) ───────
# Neural-network person segmentation — works when sitting still.
# When running from source: selfie_segmenter.tflite is downloaded automatically
# on first use and cached next to the script for future runs.
# When running as a frozen exe: the model is bundled inside the exe via the
# PyInstaller spec and loaded directly from sys._MEIPASS — no download needed.
class Segmentor:
    def __init__(self):
        self._seg  = None   # created lazily on first use
        self._ready = False
        self.mask  = None

    def _ensure_loaded(self):
        """Load the segmentation model on first use (lazy init).

        From source: downloads selfie_segmenter.tflite automatically if missing,
        then caches it next to the script for future runs.
        From frozen exe: the model is bundled in sys._MEIPASS and always found —
        no internet connection is required.
        """
        if self._ready:
            return True
        try:
            _model = _resource_path('selfie_segmenter.tflite')
            if not os.path.exists(_model):
                print('[Segmentor] Model not found — downloading selfie_segmenter.tflite …')
                import urllib.request
                _URL = (
                    'https://storage.googleapis.com/mediapipe-models/'
                    'image_segmenter/selfie_segmenter/float16/latest/'
                    'selfie_segmenter.tflite'
                )
                urllib.request.urlretrieve(_URL, _model)
                print('[Segmentor] Download complete.')
            from mediapipe.tasks import python as _mpt
            from mediapipe.tasks.python import vision as _mpv
            opts = _mpv.ImageSegmenterOptions(
                base_options=_mpt.BaseOptions(model_asset_path=_model),
                running_mode=_mpv.RunningMode.IMAGE,
                output_category_mask=False,
                output_confidence_masks=True)
            self._seg   = _mpv.ImageSegmenter.create_from_options(opts)
            self._ready = True
            return True
        except Exception as e:
            print(f'[Segmentor] Failed to load model: {e}')
            return False

    def process(self, bgr_frame):
        if not self._ensure_loaded():
            return None
        # MediaPipe Tasks API expects RGB mp.Image
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._seg.segment(mp_img)
        # confidence_masks[0] shape: (H, W, 1) float32, 1=person 0=background
        self.mask = result.confidence_masks[0].numpy_view()[:, :, 0]
        return self.mask

    def composite(self, frame, mask, mode, bg_color, bg_image, blur_amount, threshold, edge_smooth):
        if mask is None:
            return frame
        H, W = frame.shape[:2]
        # Resize mask to frame size if needed
        m = cv2.resize(mask, (W, H)) if mask.shape[:2] != (H, W) else mask.copy()
        # Apply threshold
        m = np.where(m > threshold, 1.0, 0.0).astype(np.float32)
        # Smooth edges
        if edge_smooth > 0:
            ks = edge_smooth * 2 + 1
            m = cv2.GaussianBlur(m, (ks, ks), 0)
        # m=1 means person (foreground), m=0 means background
        person_alpha = m[:, :, None]
        bg_alpha     = 1.0 - person_alpha

        if mode == 'blur':
            bs = blur_amount * 2 + 1
            bg = cv2.GaussianBlur(frame, (bs, bs), 0)
        elif mode == 'color':
            bg = np.full_like(frame, bg_color)
        elif mode == 'replace' and bg_image is not None:
            bg = cv2.resize(bg_image, (W, H))
        else:
            return frame

        return np.clip(
            frame.astype(np.float32) * person_alpha +
            bg.astype(np.float32)    * bg_alpha,
            0, 255
        ).astype(np.uint8)

# ── Main Window ───────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Camera Effects Playground')
        self.setMinimumSize(1100, 700)

        # State
        self.capture_thread = None
        self.segmentor      = Segmentor()
        self.active_effect  = 'none'
        self.bg_mode        = 'none'
        self._video_writer  = None
        self._recording     = False
        self.bg_color       = np.array([0, 255, 0], dtype=np.uint8)  # green BGR
        self.bg_image       = None
        self.blur_amount    = 15
        self.threshold      = 0.30
        self.edge_smooth    = 5
        self.effect_state   = {
            'glitchIntensity': 50, 'pixelSize': 12, 'chromaticShift': 6,
            'dreamBloom': 50, 'asciiRes': 8, 'twistAmount': 50,
            'waveAmplitude': 20, 'waveFrequency': 5, 'kaleidoSegments': 6,
            'waterStrength': 30, 'waterSnap': 50,
            'rotoSpeed': 15, 'rotoZoom': 150, 'cubeSpeed': 20,
            'snowGhost': 30,
            'embossDepth': 50,
            'hamster2Scale': 150,
            'angryIntensity': 60,
            'sidShift': 15,
            'slimeCount': 3, 'slimeSpeed': 40,
        }
        self.fps_frames = 0
        self.fps_time   = time.time()
        self.fps_val    = 0

        # Set window & taskbar icon from the camera emoji rendered to a pixmap
        icon_pix = QPixmap(64, 64)
        icon_pix.fill(Qt.GlobalColor.transparent)
        from PyQt6.QtGui import QPainter
        painter = QPainter(icon_pix)
        painter.setFont(QFont('Segoe UI Emoji', 40))
        painter.drawText(icon_pix.rect(), Qt.AlignmentFlag.AlignCenter, '🎥')
        painter.end()
        self.setWindowIcon(QIcon(icon_pix))

        self._build_ui()
        self._populate_cameras()
        self.resize(1310, 900)
        self.show()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Left: effects panel
        root.addWidget(self._build_effects_panel(), 0)
        # Center: preview
        root.addWidget(self._build_preview_panel(), 1)
        # Right: controls
        root.addWidget(self._build_controls_panel(), 0)

    # ── Effects panel (left) ─────────────────────────────────────
    def _build_effects_panel(self):
        panel = QGroupBox('✨ Special Effects')
        panel.setFixedWidth(290)
        layout = QVBoxLayout(panel)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QWidget()
        grid  = QGridLayout(inner)
        grid.setSpacing(4)
        grid.setContentsMargins(2, 2, 2, 2)

        self.effect_btns = {}
        for i, (key, info) in enumerate(EFFECTS.items()):
            btn = QPushButton(f"{info['icon']}\n{info['label']}")
            btn.setCheckable(True)
            btn.setChecked(key == 'none')
            if key == 'none':
                btn.setStyleSheet('font-weight: bold; color: #00cfff;')
            btn.setFixedHeight(52)
            btn.clicked.connect(lambda checked, k=key: self._select_effect(k))
            self.effect_btns[key] = btn
            grid.addWidget(btn, i // 2, i % 2)

        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        # Fixed-height controls area — always reserves space so buttons never move.
        self.effect_controls_box = QGroupBox('Effect Controls')
        self.effect_controls_box.setFixedHeight(110)
        self.effect_controls_box.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.effect_controls_layout = QVBoxLayout(self.effect_controls_box)
        self.effect_controls_layout.setSpacing(4)
        self.effect_controls_layout.setContentsMargins(6, 6, 6, 6)
        self.effect_controls_layout.addStretch(1)
        layout.addWidget(self.effect_controls_box, 0)

        return panel

    # ── Preview panel (center) ───────────────────────────────────
    def _build_preview_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet('background: #111;')
        layout.addWidget(self.preview_label, 1)

        footer = QHBoxLayout()

        # Left: FPS counter
        self.fps_label = QLabel('FPS: 0')
        self.fps_label.setStyleSheet('color: #aaa;')
        self.fps_label.setFixedWidth(60)

        # Center: save/status messages
        self.save_msg_label = QLabel('')
        self.save_msg_label.setStyleSheet('color: #4caf50; font-size: 12px;')
        self.save_msg_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)

        # Right: credits
        credit_label = QLabel('© 2026 Jeff Molofee (NeHe)')
        credit_label.setStyleSheet('color: #555; font-size: 11px;')
        credit_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        credit_label.setFixedWidth(180)

        footer.addWidget(self.fps_label)
        footer.addStretch()
        footer.addWidget(self.save_msg_label)
        footer.addStretch()
        footer.addWidget(credit_label)
        layout.addLayout(footer)
        return panel

    # ── Controls panel (right) ───────────────────────────────────
    def _build_controls_panel(self):
        panel = QWidget()
        panel.setFixedWidth(240)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # Camera selection
        cam_box = QGroupBox('📷 Camera')
        cam_layout = QVBoxLayout(cam_box)
        self.cam_combo = QComboBox()
        self.start_btn = QPushButton('▶ Start Camera')
        self.stop_btn  = QPushButton('⏹ Stop Camera')
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_camera)
        self.stop_btn.clicked.connect(self._stop_camera)
        cam_layout.addWidget(self.cam_combo)
        cam_layout.addWidget(self.start_btn)
        cam_layout.addWidget(self.stop_btn)
        self.cam_status = QLabel('● Camera Off')
        self.cam_status.setStyleSheet('color: red;')
        cam_layout.addWidget(self.cam_status)
        layout.addWidget(cam_box)

        # Background mode
        bg_box = QGroupBox('🎨 Background Mode')
        bg_layout = QGridLayout(bg_box)
        self.bg_btns = {}
        for i, (mode, lbl) in enumerate([('none','Normal'),('blur','Blur'),('color','BG Color'),('replace','BG Image')]):
            b = QPushButton(lbl)
            b.setCheckable(True)
            b.clicked.connect(lambda checked, m=mode: self._set_bg_mode(m))
            self.bg_btns[mode] = b
            bg_layout.addWidget(b, i // 2, i % 2)
        layout.addWidget(bg_box)
        # Apply initial active styling (Normal is selected by default)
        self._set_bg_mode('none')

        # Blur slider
        blur_box = QGroupBox('🌫 Blur Intensity')
        blur_layout = QVBoxLayout(blur_box)
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(1, 40); self.blur_slider.setValue(15)
        self.blur_lbl = QLabel('15 px')
        self.blur_slider.valueChanged.connect(lambda v: (setattr(self,'blur_amount',v), self.blur_lbl.setText(f'{v} px')))
        blur_layout.addWidget(self.blur_lbl)
        blur_layout.addWidget(self.blur_slider)
        layout.addWidget(blur_box)

        # BG Color picker
        color_box = QGroupBox('🎨 BG Color')
        color_layout = QHBoxLayout(color_box)
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(32, 32)
        self.color_preview.setStyleSheet('background: #00ff00; border: 1px solid #555;')
        pick_btn = QPushButton('Pick…')
        pick_btn.clicked.connect(self._pick_color)
        color_layout.addWidget(self.color_preview)
        color_layout.addWidget(pick_btn)
        layout.addWidget(color_box)

        # BG Image
        img_box = QGroupBox('🖼 BG Image')
        img_layout = QVBoxLayout(img_box)
        load_btn = QPushButton('Load Image…')
        load_btn.clicked.connect(self._load_bg_image)
        img_layout.addWidget(load_btn)
        layout.addWidget(img_box)

        # Segmentation settings
        seg_box = QGroupBox('⚙ Segmentation')
        seg_layout = QVBoxLayout(seg_box)
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(5, 90); self.thresh_slider.setValue(30)
        self.thresh_lbl = QLabel('Threshold: 0.30')
        self.thresh_slider.valueChanged.connect(lambda v: (setattr(self,'threshold',v/100), self.thresh_lbl.setText(f'Threshold: {v/100:.2f}')))
        self.edge_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_slider.setRange(0, 10); self.edge_slider.setValue(5)
        self.edge_lbl = QLabel('Edge Smooth: 5')
        self.edge_slider.valueChanged.connect(lambda v: (setattr(self,'edge_smooth',v), self.edge_lbl.setText(f'Edge Smooth: {v}')))
        seg_layout.addWidget(self.thresh_lbl)
        seg_layout.addWidget(self.thresh_slider)
        seg_layout.addWidget(self.edge_lbl)
        seg_layout.addWidget(self.edge_slider)
        layout.addWidget(seg_box)

        # Capture group — Snapshot + Record, naturally placed after segmentation
        cap_box = QGroupBox('📁 Capture')
        cap_layout = QVBoxLayout(cap_box)
        cap_layout.setSpacing(6)

        self.snap_btn = QPushButton('📸  Take Snapshot')
        self.snap_btn.setFixedHeight(34)
        self.snap_btn.clicked.connect(self._take_snapshot)
        cap_layout.addWidget(self.snap_btn)

        self.record_btn = QPushButton('⏺  Record Video')
        self.record_btn.setCheckable(True)
        self.record_btn.setFixedHeight(34)
        self.record_btn.clicked.connect(self._toggle_recording)
        cap_layout.addWidget(self.record_btn)

        layout.addWidget(cap_box)
        layout.addStretch()
        return panel

    # ── Camera methods ───────────────────────────────────────────
    def _populate_cameras(self):
        self.cam_combo.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cam_combo.addItem(f'Camera {i}', i)
                cap.release()
        if self.cam_combo.count() > 0:
            self._start_camera()

    def _start_camera(self):
        self._stop_camera()
        idx = self.cam_combo.currentData()
        if idx is None:
            return
        self.capture_thread = CaptureThread(device_id=idx)
        self.capture_thread.frame_ready.connect(self._on_frame)
        self.capture_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.cam_status.setText('● Camera On')
        self.cam_status.setStyleSheet('color: lime;')

    def _stop_camera(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.cam_status.setText('● Camera Off')
        self.cam_status.setStyleSheet('color: red;')

    # ── Frame processing ─────────────────────────────────────────
    def _on_frame(self, frame):
        # frame_pending must be cleared no matter what happens below —
        # an uncaught exception here would otherwise leave it stuck True
        # forever, permanently stalling the capture thread (it only
        # emits when frame_pending is False). That would freeze the
        # feed entirely, which is worse than the backlog this flag
        # exists to prevent.
        try:
            # Background compositing
            if self.bg_mode != 'none':
                mask = self.segmentor.process(frame)
                frame = self.segmentor.composite(
                    frame, mask, self.bg_mode,
                    self.bg_color, self.bg_image,
                    self.blur_amount, self.threshold, self.edge_smooth,
                )

            # Apply effect
            fx_info = EFFECTS.get(self.active_effect)
            if fx_info and self.active_effect != 'none':
                try:
                    frame = fx_info['fn'](frame, self.effect_state)
                except Exception:
                    pass

            # Update FPS
            self.fps_frames += 1
            now = time.time()
            if now - self.fps_time >= 1.0:
                self.fps_val    = self.fps_frames
                self.fps_frames = 0
                self.fps_time   = now
                self.fps_label.setText(f'FPS: {self.fps_val}')

            # Record frame if active
            if self._recording and self._video_writer is not None:
                self._video_writer.write(frame)

            # Display
            self._show_frame(frame)
        finally:
            # Let the capture thread know it's safe to emit the next frame.
            if self.capture_thread:
                self.capture_thread.frame_pending = False

    def _show_frame(self, frame):
        self._last_frame = frame
        H, W = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = QImage(rgb.data, W, H, W * 3, QImage.Format.Format_RGB888)
        pix   = QPixmap.fromImage(img)
        lw, lh = self.preview_label.width(), self.preview_label.height()
        scaled = pix.scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            Qt.TransformationMode.SmoothTransformation)
        # Crop to label size so it fills top-to-bottom, clipping sides if needed
        x_off = (scaled.width()  - lw) // 2
        y_off = (scaled.height() - lh) // 2
        self.preview_label.setPixmap(scaled.copy(x_off, y_off, lw, lh))

    # ── UI event handlers ────────────────────────────────────────
    def _select_effect(self, key):
        self.active_effect = key
        for k, btn in self.effect_btns.items():
            if k == key:
                btn.setChecked(True)
                btn.setStyleSheet('font-weight: bold; color: #00cfff;')
            else:
                btn.setChecked(False)
                btn.setStyleSheet('')

        # If an effect is turned on, force background mode back to none
        if key != 'none' and self.bg_mode != 'none':
            self._set_bg_mode('none')

        # Clear old controls (widgets only; spacer items removed too)
        while self.effect_controls_layout.count():
            item = self.effect_controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Build new controls, then push remaining space to bottom via stretch
        info = EFFECTS[key]
        for ctrl in info.get('controls', []):
            state_key, label, unit, lo, hi, default, parse = ctrl
            if parse == 'bool':
                # Render as a checkbox (lo/hi/unit ignored)
                from PyQt6.QtWidgets import QCheckBox
                cb = QCheckBox(label)
                cb.setChecked(bool(self.effect_state.get(state_key, default)))
                def _cb_changed(checked, sk=state_key):
                    self.effect_state[sk] = int(checked)
                cb.checkStateChanged.connect(lambda state, sk=state_key: self.effect_state.update({sk: int(state == Qt.CheckState.Checked)}))
                self.effect_controls_layout.addWidget(cb)
            else:
                lbl = QLabel(f'{label}: {self.effect_state.get(state_key, default)}{unit}')
                sld = QSlider(Qt.Orientation.Horizontal)
                sld.setRange(lo, hi)
                sld.setValue(self.effect_state.get(state_key, default))
                def _changed(v, sk=state_key, lb=lbl, u=unit, la=label):
                    self.effect_state[sk] = v
                    lb.setText(f'{la}: {v}{u}')
                sld.valueChanged.connect(_changed)
                self.effect_controls_layout.addWidget(lbl)
                self.effect_controls_layout.addWidget(sld)
        self.effect_controls_layout.addStretch(1)  # always pin content to top

    def _set_bg_mode(self, mode):
        self.bg_mode = mode
        for m, btn in self.bg_btns.items():
            if m == mode:
                btn.setChecked(True)
                btn.setStyleSheet('font-weight: bold; color: #00cfff;')
            else:
                btn.setChecked(False)
                btn.setStyleSheet('')
        # If a real bg mode is selected, switch effect back to none
        if mode != 'none' and self.active_effect != 'none':
            self._select_effect('none')

    def _pick_color(self):
        from PyQt6.QtWidgets import QColorDialog
        c = QColorDialog.getColor(QColor(0, 255, 0), self)
        if c.isValid():
            self.bg_color = np.array([c.blue(), c.green(), c.red()], dtype=np.uint8)
            self.color_preview.setStyleSheet(f'background: {c.name()}; border: 1px solid #555;')

    def _load_bg_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Background Image', '',
                                               'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.bg_image = img
                self._set_bg_mode('replace')

    def _show_save_msg(self, msg, color='#4caf50'):
        """Show a save confirmation message below the preview, auto-clears after 4s."""
        self.save_msg_label.setText(msg)
        self.save_msg_label.setStyleSheet(f'color: {color}; font-size: 12px;')
        QTimer.singleShot(4000, lambda: self.save_msg_label.setText(''))

    def _take_snapshot(self):
        if not hasattr(self, '_last_frame') or self._last_frame is None:
            return
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'snapshot_{ts}.png'
        cv2.imwrite(path, self._last_frame)
        self._show_save_msg(f'📸 Snapshot saved: {path}')

    def _toggle_recording(self):
        if not self._recording:
            # Start recording
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f'recording_{ts}.avi'
            frame = getattr(self, '_last_frame', None)
            H, W = (frame.shape[:2] if frame is not None else (720, 1280))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self._video_writer = cv2.VideoWriter(path, fourcc, 30.0, (W, H))
            self._recording    = True
            self._record_path  = path
            self.record_btn.setText('⏹ Stop Recording')
            self.record_btn.setStyleSheet('color: red; font-weight: bold;')
            self._show_save_msg(f'⏺ Recording: {path}', color='#f44336')
        else:
            # Stop recording
            self._recording = False
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None
            self.record_btn.setText('⏺ Record Video')
            self.record_btn.setChecked(False)
            self.record_btn.setStyleSheet('')
            self._show_save_msg(f'🎬 Video saved: {self._record_path}')

    def closeEvent(self, event):
        if self._recording:
            self._toggle_recording()
        self._stop_camera()
        super().closeEvent(event)

# ── Entry point ───────────────────────────────────────────────────
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    # Dark palette
    palette = app.palette()
    from PyQt6.QtGui import QPalette
    palette.setColor(QPalette.ColorRole.Window,          QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base,            QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button,          QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
