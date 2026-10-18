"""Finish desktop window."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.finish import (
    MAX_REPEATS,
    MIN_REPEATS,
    FinishResult,
    FinishSettings,
    PreparedSurface,
    displace,
    edge_target,
    estimate_triangles,
    export_stl,
    load_mesh,
    prepare_surface,
    sample_cube,
)
from app.patterns import BUILT_IN, build_patterns, load_image, swatch
from app.theme import STYLESHEET
from app.viewport import FinishViewport

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_DEPTH = 0.3


def _fmt(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _pixmap(rgb: np.ndarray) -> QPixmap:
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class _FitLabel(QLabel):
    """Word-wrapped label that follows the column instead of widening it."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        if width < 1:
            return self.fontMetrics().lineSpacing()
        flags = int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap)
        rect = self.fontMetrics().boundingRect(0, 0, width, 10000, flags, self.text())
        return max(rect.height(), self.fontMetrics().lineSpacing())

    def sizeHint(self) -> QSize:  # noqa: N802
        metrics = self.fontMetrics()
        natural = metrics.horizontalAdvance(self.text()) + 2
        # Paragraphs wrap inside the column. A hint wider than the fixed
        # controls would make the scroll body look larger than the panel.
        width = min(natural, 220) if natural > 0 else metrics.lineSpacing()
        return QSize(max(width, 1), self.heightForWidth(max(width, 1)))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, self.fontMetrics().lineSpacing())


class _ElideLabel(QLabel):
    """Single-line title that shrinks by eliding instead of widening the column."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setMinimumWidth(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802
        self._full = text
        self._apply()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply()

    def _apply(self) -> None:
        width = self.width() if self.width() > 8 else 160
        elided = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, width)
        if self.text() != elided:
            QLabel.setText(self, elided)

    def sizeHint(self) -> QSize:  # noqa: N802
        height = max(super().sizeHint().height(), self.fontMetrics().lineSpacing())
        width = min(self.fontMetrics().horizontalAdvance(self._full) + 4, 160)
        return QSize(max(width, 64), height)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(64, self.fontMetrics().lineSpacing())


class _CarveWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(int, str)

    def __init__(
        self,
        job_id: int,
        vertices: np.ndarray,
        faces: np.ndarray,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        prepared: PreparedSurface | None,
        image: np.ndarray,
        settings: FinishSettings,
        max_edge: float,
        clamped: bool,
        cache_key: tuple,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.vertices = np.array(vertices, dtype=np.float64, copy=True)
        self.faces = np.array(faces, dtype=np.int64, copy=True)
        self.bounds_min = np.array(bounds_min, dtype=np.float64, copy=True)
        self.bounds_max = np.array(bounds_max, dtype=np.float64, copy=True)
        self.prepared = prepared
        self.image = np.array(image, dtype=np.float32, copy=True)
        self.settings = settings
        self.max_edge = max_edge
        self.clamped = clamped
        self.cache_key = cache_key

    def run(self) -> None:
        try:
            prepared = self.prepared
            if prepared is None:
                prepared = prepare_surface(self.vertices, self.faces, self.max_edge, self.clamped)
            result = displace(
                prepared,
                self.image,
                self.settings,
                self.bounds_min,
                self.bounds_max,
            )
            self.succeeded.emit(
                {
                    "job_id": self.job_id,
                    "prepared": prepared,
                    "result": result,
                    "cache_key": self.cache_key,
                }
            )
        except Exception as exc:
            self.failed.emit(self.job_id, str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STL Texture Applier")
        self.resize(1240, 820)
        self.setMinimumSize(980, 680)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLESHEET)

        self.patterns = build_patterns()
        self.pattern_name = "Wood"
        self.custom_height: np.ndarray | None = None
        self.custom_label = "Image"
        self.height = self.patterns["Wood"]

        self.source_vertices: np.ndarray | None = None
        self.source_faces: np.ndarray | None = None
        self.bounds_min: np.ndarray | None = None
        self.bounds_max: np.ndarray | None = None
        self.model_size = 1.0
        self.model_area = 1.0
        self.model_title = ""
        self.export_stem = "finish"
        self.mesh_id = 0
        self.cache: dict[tuple, PreparedSurface] = {}
        self.result: FinishResult | None = None
        self.job_id = 0
        self.worker: _CarveWorker | None = None
        self.pending = False
        self._guard = False
        self._split_applied = False

        self._build()
        self._build_menu()
        QTimer.singleShot(0, self._load_sample)

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 16)
        outer.setSpacing(0)

        panel = QFrame()
        panel.setObjectName("panel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._panel = panel
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 16, 16, 14)
        panel_layout.setSpacing(10)

        title = QLabel("STL Texture Applier")
        title.setObjectName("title")
        self._title = title
        subtitle = QLabel("Carve a texture into an STL.")
        subtitle.setObjectName("subtitle")
        panel_layout.addWidget(title)
        panel_layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body = QWidget()
        body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        form = QVBoxLayout(body)
        form.setContentsMargins(0, 8, 8, 8)
        form.setSpacing(8)
        scroll.setWidget(body)
        self._scroll = scroll
        self._body = body
        self._form = form
        panel_layout.addWidget(scroll, 1)

        form.addWidget(self._section("Model"))
        self.model_name = _FitLabel("Sample cube")
        self.model_meta = _FitLabel("")
        self.model_meta.setObjectName("muted")
        form.addWidget(self.model_name)
        form.addWidget(self.model_meta)
        model_row = QHBoxLayout()
        open_stl = QPushButton("Open STL")
        open_stl.clicked.connect(self._open_stl)
        sample = QPushButton("Sample cube")
        sample.clicked.connect(self._load_sample)
        model_row.addWidget(open_stl)
        model_row.addWidget(sample)
        form.addLayout(model_row)

        form.addSpacing(8)
        form.addWidget(self._section("Texture"))
        self.pattern_group = QButtonGroup(self)
        self.pattern_group.setExclusive(True)
        self.pattern_group.idClicked.connect(self._on_pattern)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.pattern_buttons: dict[int, QToolButton] = {}
        for column in range(3):
            grid.setColumnStretch(column, 1)
        for index, (name, tip, _fn) in enumerate(BUILT_IN):
            button = self._pattern_button(name, tip, swatch(self.patterns[name]))
            self.pattern_group.addButton(button, index)
            self.pattern_buttons[index] = button
            grid.addWidget(button, index // 3, index % 3)
        blank = np.full((72, 72, 3), 34, dtype=np.uint8)
        blank[33:39, 22:50] = 198
        blank[22:50, 33:39] = 198
        self.image_button = self._pattern_button("Image", "Use your own picture", blank)
        self.pattern_group.addButton(self.image_button, len(BUILT_IN))
        grid.addWidget(self.image_button, len(BUILT_IN) // 3, len(BUILT_IN) % 3)
        form.addLayout(grid)
        self.pattern_buttons[0].setChecked(True)
        browse = QPushButton("Open image")
        browse.clicked.connect(self._open_image)
        form.addWidget(browse)

        form.addSpacing(8)
        form.addWidget(self._section("Surface"))
        self.depth_slider, self.depth_value, self.depth_spin = self._slider_block(
            form, "Depth", 0, 1000, self._on_depth_slider, self._on_depth_spin
        )
        self.scale_slider, self.scale_value, self.scale_spin = self._slider_block(
            form, "Scale", 0, 1000, self._on_scale_slider, self._on_scale_spin
        )
        self.detail_slider, self.detail_value = self._plain_slider(
            form, "Detail", 0, 1000, self._on_detail_slider
        )
        self.detail_caption = _FitLabel("")
        self.detail_caption.setObjectName("muted")
        form.addWidget(self.detail_caption)

        self.contrast_slider, self.contrast_value, self.contrast_spin = self._slider_block(
            form, "Contrast", 0, 1000, self._on_contrast_slider, self._on_contrast_spin
        )
        self.angle_slider, self.angle_value, self.angle_spin = self._slider_block(
            form, "Angle", 0, 1000, self._on_angle_slider, self._on_angle_spin
        )
        self.shift_x_slider, self.shift_x_value, self.shift_x_spin = self._slider_block(
            form, "Shift sideways", 0, 1000, self._on_shift_x_slider, self._on_shift_x_spin
        )
        self.shift_y_slider, self.shift_y_value, self.shift_y_spin = self._slider_block(
            form, "Shift up", 0, 1000, self._on_shift_y_slider, self._on_shift_y_spin
        )

        self.invert = QCheckBox("Invert the pattern")
        self.invert.toggled.connect(self._on_option)
        self.centered = QCheckBox("Keep the part the same overall size")
        self.centered.setChecked(True)
        self.centered.toggled.connect(self._on_option)
        form.addWidget(self.invert)
        form.addWidget(self.centered)
        self.hint = _FitLabel("")
        self.hint.setObjectName("hint")
        form.addWidget(self.hint)
        note = _FitLabel(
            "The finish is carved into the surface and stays in the STL. The underside is left smooth so the part sits flat on the bed."
        )
        note.setObjectName("hint")
        form.addWidget(note)
        form.addStretch(1)

        self.output_button = QPushButton("Output STL")
        self.output_button.setObjectName("primary")
        self.output_button.setEnabled(False)
        self.output_button.clicked.connect(self._export)
        panel_layout.addWidget(self.output_button)
        self.status = _FitLabel("Opening a sample cube.")
        self.status.setObjectName("status")
        panel_layout.addWidget(self.status)

        stage_host = QWidget()
        stage_host.setObjectName("stage")
        stage_host.setMinimumWidth(220)
        stage_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._stage = stage_host
        stage = QVBoxLayout(stage_host)
        stage.setContentsMargins(0, 0, 0, 0)
        stage.setSpacing(8)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self._toolbar = toolbar
        self.view_title = _ElideLabel("Sample cube")
        self.view_title.setObjectName("title")
        toolbar.addWidget(self.view_title, 1)
        self.frame_button = QPushButton("Frame")
        self.frame_button.setObjectName("ghost")
        self.frame_button.clicked.connect(self._frame)
        self.shaded_button = QPushButton("Shaded")
        self.shaded_button.setObjectName("ghost")
        self.shaded_button.setCheckable(True)
        self.shaded_button.setChecked(True)
        self.wire_button = QPushButton("Wireframe")
        self.wire_button.setObjectName("ghost")
        self.wire_button.setCheckable(True)
        self.shaded_button.clicked.connect(lambda: self._set_wire(False))
        self.wire_button.clicked.connect(lambda: self._set_wire(True))
        toolbar.addWidget(self.frame_button)
        toolbar.addWidget(self.shaded_button)
        toolbar.addWidget(self.wire_button)
        stage.addLayout(toolbar)

        frame = QFrame()
        frame.setObjectName("viewportFrame")
        frame.setMinimumWidth(0)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(1, 1, 1, 1)
        self.viewport = FinishViewport(frame)
        self.viewport.file_dropped.connect(self._take_path)
        frame_layout.addWidget(self.viewport)
        stage.addWidget(frame, 1)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFixedHeight(3)
        stage.addWidget(self.progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("split")
        splitter.setChildrenCollapsible(False)
        splitter.setOpaqueResize(True)
        splitter.setHandleWidth(14)
        splitter.addWidget(panel)
        splitter.addWidget(stage_host)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        self._splitter = splitter
        outer.addWidget(splitter)
        QTimer.singleShot(0, self._finish_layout)

        self._set_scale(2.5)
        self._set_detail(0.55)
        self._set_contrast(1.15)
        self._set_angle(0.0)
        self._set_shift_x(0.0)
        self._set_shift_y(0.0)
        self._refresh_hint()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_stl = QAction("Open STL…", self)
        open_stl.setShortcut(QKeySequence.StandardKey.Open)
        open_stl.triggered.connect(self._open_stl)
        open_image = QAction("Open image…", self)
        open_image.setShortcut(QKeySequence("Ctrl+I"))
        open_image.triggered.connect(self._open_image)
        output = QAction("Output STL…", self)
        output.setShortcut(QKeySequence("Ctrl+E"))
        output.triggered.connect(self._export)
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(open_stl)
        file_menu.addAction(open_image)
        file_menu.addSeparator()
        file_menu.addAction(output)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

    def _section(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("section")
        return label

    def _pattern_button(self, name: str, tip: str, rgb: np.ndarray | None) -> QToolButton:
        button = QToolButton()
        button.setText(name)
        button.setToolTip(tip)
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        button.setIconSize(QSize(64, 64))
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setMinimumHeight(92)
        if rgb is not None:
            button.setIcon(QIcon(_pixmap(rgb)))
        return button

    def _slider_block(self, form, title, low, high, on_slider, on_spin):
        header = QHBoxLayout()
        name = QLabel(title)
        value = QLabel("")
        value.setObjectName("value")
        header.addWidget(name)
        header.addStretch(1)
        header.addWidget(value)
        form.addLayout(header)
        row = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.valueChanged.connect(on_slider)
        spin = QDoubleSpinBox()
        spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        spin.valueChanged[float].connect(on_spin)
        self._fit_spin(spin)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        form.addLayout(row)
        return slider, value, spin

    def _plain_slider(self, form, title, low, high, on_slider):
        header = QHBoxLayout()
        name = QLabel(title)
        value = QLabel("")
        value.setObjectName("value")
        header.addWidget(name)
        header.addStretch(1)
        header.addWidget(value)
        form.addLayout(header)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.valueChanged.connect(on_slider)
        form.addWidget(slider)
        return slider, value

    def _settings(self) -> FinishSettings:
        return FinishSettings(
            depth=float(self.depth_spin.value()),
            scale=float(self.scale_spin.value()),
            detail=self.detail_slider.value() / 1000.0,
            contrast=float(self.contrast_spin.value()),
            rotation=float(self.angle_spin.value()),
            offset_x=float(self.shift_x_spin.value()),
            offset_y=float(self.shift_y_spin.value()),
            invert=self.invert.isChecked(),
            centered=self.centered.isChecked(),
        )

    def _refresh_hint(self) -> None:
        if self.centered.isChecked() and not self.invert.isChecked():
            text = "Light areas rise off the surface. Dark areas sink in. Mid-gray stays put."
        elif self.centered.isChecked():
            text = "Dark areas rise off the surface. Light areas sink in. Mid-gray stays put."
        elif self.invert.isChecked():
            text = "Darker areas stand farther off the original surface."
        else:
            text = "Brighter areas stand farther off the original surface."
        self.hint.setText(text)

    def _refresh_detail_caption(self) -> None:
        if self.source_faces is None:
            self.detail_caption.setText("")
            return
        settings = self._settings()
        max_edge, clamped = edge_target(
            self.model_size, self.model_area, settings.scale, settings.detail
        )
        estimate = estimate_triangles(self.model_area, max_edge, len(self.source_faces))
        if settings.detail < 0.34:
            word = "Coarse"
        elif settings.detail < 0.67:
            word = "Medium"
        else:
            word = "Fine"
        suffix = " · limited so the file stays workable" if clamped else ""
        self.detail_caption.setText(f"{word} · about {estimate:,} triangles{suffix}")

    def _set_depth_range(self, reset: bool) -> None:
        depth_max = max(self.model_size * 0.08, 1e-4)
        self._guard = True
        self.depth_spin.setRange(0.0, depth_max)
        self.depth_spin.setDecimals(3 if depth_max < 10 else 2)
        self.depth_spin.setSingleStep(depth_max / 100.0)
        self._guard = False
        if reset:
            self._set_depth(min(DEFAULT_DEPTH, depth_max))
        self._fit_spin(self.depth_spin)
        self._lock_panel_minimum()

    def _set_depth(self, depth: float) -> None:
        depth_max = max(self.depth_spin.maximum(), 1e-9)
        slider = int(round(np.clip(depth / depth_max, 0, 1) * 1000))
        self._guard = True
        self.depth_spin.setValue(depth)
        self.depth_slider.setValue(slider)
        self.depth_value.setText(_fmt(depth))
        self._guard = False

    def _on_depth_slider(self, value: int) -> None:
        if self._guard:
            return
        depth = self.depth_spin.maximum() * (value / 1000.0)
        self._guard = True
        self.depth_spin.setValue(depth)
        self.depth_value.setText(_fmt(depth))
        self._guard = False
        self._schedule()

    def _on_depth_spin(self, value: float) -> None:
        if self._guard:
            return
        depth_max = max(self.depth_spin.maximum(), 1e-9)
        self._guard = True
        self.depth_slider.setValue(int(round(np.clip(value / depth_max, 0, 1) * 1000)))
        self.depth_value.setText(_fmt(value))
        self._guard = False
        self._schedule()

    def _set_scale(self, repeats: float) -> None:
        repeats = float(np.clip(repeats, MIN_REPEATS, MAX_REPEATS))
        t = math.log(repeats / MIN_REPEATS) / math.log(MAX_REPEATS / MIN_REPEATS)
        self.scale_spin.setRange(MIN_REPEATS, MAX_REPEATS)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setSingleStep(0.1)
        self._guard = True
        self.scale_spin.setValue(repeats)
        self.scale_slider.setValue(int(round(t * 1000)))
        self.scale_value.setText(f"{repeats:.2f} repeats")
        self._guard = False

    def _on_scale_slider(self, value: int) -> None:
        if self._guard:
            return
        t = value / 1000.0
        repeats = MIN_REPEATS * ((MAX_REPEATS / MIN_REPEATS) ** t)
        self._guard = True
        self.scale_spin.setValue(repeats)
        self.scale_value.setText(f"{repeats:.2f} repeats")
        self._guard = False
        self._refresh_detail_caption()
        self._schedule()

    def _on_scale_spin(self, value: float) -> None:
        if self._guard:
            return
        t = math.log(max(value, MIN_REPEATS) / MIN_REPEATS) / math.log(MAX_REPEATS / MIN_REPEATS)
        self._guard = True
        self.scale_slider.setValue(int(round(np.clip(t, 0, 1) * 1000)))
        self.scale_value.setText(f"{value:.2f} repeats")
        self._guard = False
        self._refresh_detail_caption()
        self._schedule()

    def _set_detail(self, detail: float) -> None:
        self._guard = True
        self.detail_slider.setValue(int(round(np.clip(detail, 0, 1) * 1000)))
        self._guard = False
        self._label_detail()

    def _label_detail(self) -> None:
        detail = self.detail_slider.value() / 1000.0
        if detail < 0.34:
            word = "Coarse"
        elif detail < 0.67:
            word = "Medium"
        else:
            word = "Fine"
        self.detail_value.setText(word)
        self._refresh_detail_caption()

    def _on_detail_slider(self, _value: int) -> None:
        if self._guard:
            return
        self._label_detail()
        self._schedule()

    def _set_contrast(self, contrast: float) -> None:
        contrast = float(np.clip(contrast, 0.4, 2.8))
        t = (contrast - 0.4) / (2.8 - 0.4)
        self.contrast_spin.setRange(0.4, 2.8)
        self.contrast_spin.setDecimals(2)
        self.contrast_spin.setSingleStep(0.05)
        self._guard = True
        self.contrast_spin.setValue(contrast)
        self.contrast_slider.setValue(int(round(t * 1000)))
        self.contrast_value.setText(f"{contrast:.2f}")
        self._guard = False

    def _on_contrast_slider(self, value: int) -> None:
        if self._guard:
            return
        contrast = 0.4 + (value / 1000.0) * (2.8 - 0.4)
        self._guard = True
        self.contrast_spin.setValue(contrast)
        self.contrast_value.setText(f"{contrast:.2f}")
        self._guard = False
        self._schedule()

    def _on_contrast_spin(self, value: float) -> None:
        if self._guard:
            return
        t = (value - 0.4) / (2.8 - 0.4)
        self._guard = True
        self.contrast_slider.setValue(int(round(np.clip(t, 0, 1) * 1000)))
        self.contrast_value.setText(f"{value:.2f}")
        self._guard = False
        self._schedule()

    def _set_angle(self, angle: float) -> None:
        angle = float(np.clip(angle, 0.0, 180.0))
        self.angle_spin.setRange(0.0, 180.0)
        self.angle_spin.setDecimals(0)
        self.angle_spin.setSingleStep(1.0)
        self._guard = True
        self.angle_spin.setValue(angle)
        self.angle_slider.setValue(int(round(angle / 180.0 * 1000)))
        self.angle_value.setText(f"{angle:.0f}°")
        self._guard = False

    def _on_angle_slider(self, value: int) -> None:
        if self._guard:
            return
        angle = value / 1000.0 * 180.0
        self._guard = True
        self.angle_spin.setValue(angle)
        self.angle_value.setText(f"{angle:.0f}°")
        self._guard = False
        self._schedule()

    def _on_angle_spin(self, value: float) -> None:
        if self._guard:
            return
        self._guard = True
        self.angle_slider.setValue(int(round(value / 180.0 * 1000)))
        self.angle_value.setText(f"{value:.0f}°")
        self._guard = False
        self._schedule()

    def _bind_shift(self, slider, spin, label, value: float) -> None:
        value = float(np.clip(value, -1.0, 1.0))
        spin.setRange(-1.0, 1.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.01)
        self._guard = True
        spin.setValue(value)
        slider.setValue(int(round((value + 1.0) / 2.0 * 1000)))
        label.setText(f"{value:.2f}")
        self._guard = False

    def _set_shift_x(self, value: float) -> None:
        self._bind_shift(self.shift_x_slider, self.shift_x_spin, self.shift_x_value, value)

    def _set_shift_y(self, value: float) -> None:
        self._bind_shift(self.shift_y_slider, self.shift_y_spin, self.shift_y_value, value)

    def _on_shift(self, slider_value: int, spin: QDoubleSpinBox, label: QLabel) -> None:
        if self._guard:
            return
        value = -1.0 + (slider_value / 1000.0) * 2.0
        self._guard = True
        spin.setValue(value)
        label.setText(f"{value:.2f}")
        self._guard = False
        self._schedule()

    def _on_shift_spin(self, value: float, slider: QSlider, label: QLabel) -> None:
        if self._guard:
            return
        self._guard = True
        slider.setValue(int(round((value + 1.0) / 2.0 * 1000)))
        label.setText(f"{value:.2f}")
        self._guard = False
        self._schedule()

    def _on_shift_x_slider(self, value: int) -> None:
        self._on_shift(value, self.shift_x_spin, self.shift_x_value)

    def _on_shift_y_slider(self, value: int) -> None:
        self._on_shift(value, self.shift_y_spin, self.shift_y_value)

    def _on_shift_x_spin(self, value: float) -> None:
        self._on_shift_spin(value, self.shift_x_slider, self.shift_x_value)

    def _on_shift_y_spin(self, value: float) -> None:
        self._on_shift_spin(value, self.shift_y_slider, self.shift_y_value)

    def _on_option(self) -> None:
        self._refresh_hint()
        self._schedule()

    def _on_pattern(self, button_id: int) -> None:
        if button_id == len(BUILT_IN):
            if self.custom_height is None:
                self._open_image()
                if self.custom_height is None:
                    for index, (name, _tip, _fn) in enumerate(BUILT_IN):
                        if name == self.pattern_name:
                            self.pattern_buttons[index].setChecked(True)
                            break
                return
            self.pattern_name = "Image"
            self.height = self.custom_height
        else:
            self.pattern_name = BUILT_IN[button_id][0]
            self.height = self.patterns[self.pattern_name]
        self._schedule()

    def _load_sample(self) -> None:
        self._set_mesh(sample_cube(), "Sample cube", None)

    def _open_stl(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL files (*.stl)")
        if path:
            self._load_stl(path)

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open texture",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if path:
            self._load_image(path)

    def _load_stl(self, path: str) -> None:
        try:
            mesh = load_mesh(path)
        except Exception as exc:
            QMessageBox.warning(self, "STL Texture Applier", f"Couldn't read that STL.\n\n{exc}")
            return
        self._set_mesh(mesh, Path(path).name, path)

    def _load_image(self, path: str) -> None:
        try:
            height = load_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "STL Texture Applier", f"Couldn't read that image.\n\n{exc}")
            return
        self.custom_height = height
        self.custom_label = Path(path).stem
        self.image_button.setText(self._short(self.custom_label))
        self.image_button.setIcon(QIcon(_pixmap(swatch(height))))
        self.image_button.setChecked(True)
        self.pattern_name = "Image"
        self.height = height
        if float(height.std()) < 0.015:
            self.status.setText("That image is almost one flat tone, so the relief will be shallow.")
        self._schedule()

    def _set_mesh(self, mesh, title: str, path: str | None) -> None:
        self.mesh_id += 1
        self.cache.clear()
        self.result = None
        self.output_button.setEnabled(False)
        self.source_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.source_faces = np.asarray(mesh.faces, dtype=np.int64)
        self.bounds_min = np.asarray(mesh.bounds[0], dtype=np.float64)
        self.bounds_max = np.asarray(mesh.bounds[1], dtype=np.float64)
        extents = self.bounds_max - self.bounds_min
        self.model_size = float(np.max(extents))
        self.model_area = float(mesh.area)
        self.model_title = title
        self.export_stem = Path(path).stem if path else "sample_cube"
        self.model_name.setText(self._short(title, 42))
        self.view_title.setText(self._short(title, 48))
        self.model_meta.setText(
            f"{_fmt(extents[0])} × {_fmt(extents[1])} × {_fmt(extents[2])} · {len(self.source_faces):,} triangles"
        )
        self._set_depth_range(reset=True)
        self._refresh_detail_caption()
        self.viewport.show_mesh(self.source_vertices, self.source_faces, frame=True)
        self._set_wire(False)
        self.status.setText("Carving the surface.")
        self._schedule()

    def _schedule(self) -> None:
        if self.source_vertices is None:
            return
        if not hasattr(self, "_debounce"):
            self._debounce = QTimer(self)
            self._debounce.setSingleShot(True)
            self._debounce.setInterval(220)
            self._debounce.timeout.connect(self._start_job)
        self._debounce.start()

    def _start_job(self) -> None:
        if self.source_vertices is None or self.bounds_min is None or self.bounds_max is None:
            return
        if self.worker is not None and self.worker.isRunning():
            self.pending = True
            return
        settings = self._settings()
        if settings.depth <= 0 and self.source_faces is not None:
            faces = self.source_faces
            self.result = FinishResult(
                vertices=np.array(self.source_vertices, copy=True),
                faces=faces,
                triangles=int(len(faces)),
                clamped=False,
            )
            self.viewport.show_mesh(self.result.vertices, self.result.faces, frame=False)
            self.output_button.setEnabled(True)
            self._set_busy(False)
            self.status.setText(f"{self.result.triangles:,} triangles. Depth is zero, so this is the original surface.")
            return
        max_edge, clamped = edge_target(
            self.model_size, self.model_area, settings.scale, settings.detail
        )
        cache_key = (self.mesh_id, round(max_edge, 5))
        prepared = self.cache.get(cache_key)
        self.job_id += 1
        self.pending = False
        self._set_busy(True)
        self.worker = _CarveWorker(
            self.job_id,
            self.source_vertices,
            self.source_faces,
            self.bounds_min,
            self.bounds_max,
            prepared,
            self.height,
            settings,
            max_edge,
            clamped,
            cache_key,
        )
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_success(self, payload: dict) -> None:
        prepared: PreparedSurface = payload["prepared"]
        self.cache[payload["cache_key"]] = prepared
        while len(self.cache) > 4:
            oldest = next(iter(self.cache))
            if oldest == payload["cache_key"]:
                self.cache[oldest] = self.cache.pop(oldest)
                break
            self.cache.pop(oldest, None)
        if payload["job_id"] != self.job_id:
            self._after_job()
            return
        result: FinishResult = payload["result"]
        self.result = result
        self.viewport.show_mesh(result.vertices, result.faces, frame=False)
        self.output_button.setEnabled(True)
        limited = " Detail was limited so the file stays workable." if result.clamped else ""
        self.status.setText(f"{result.triangles:,} triangles.{limited}")
        self._set_busy(False)
        self._after_job()

    def _on_fail(self, job_id: int, message: str) -> None:
        if job_id == self.job_id:
            self._set_busy(False)
            self.status.setText("The surface could not be carved.")
            QMessageBox.warning(self, "STL Texture Applier", message or "The surface could not be carved.")
        self._after_job()

    def _after_job(self) -> None:
        if self.pending:
            self.pending = False
            self._start_job()

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.progress.setRange(0, 0)
            self.status.setText("Carving the surface.")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def _export(self) -> None:
        if self.result is None:
            return
        suggested = str(Path.home() / f"{self.export_stem}_finish.stl")
        path, _ = QFileDialog.getSaveFileName(self, "Output STL", suggested, "STL files (*.stl)")
        if not path:
            return
        if not path.lower().endswith(".stl"):
            path += ".stl"
        try:
            export_stl(path, self.result.vertices, self.result.faces)
        except Exception as exc:
            QMessageBox.warning(self, "STL Texture Applier", f"Couldn't save that file.\n\n{exc}")
            return
        self.status.setText(f"Saved {Path(path).name} · {self.result.triangles:,} triangles.")

    def _frame(self) -> None:
        self.viewport.frame_view()

    def _set_wire(self, enabled: bool) -> None:
        self.wire_button.setChecked(enabled)
        self.shaded_button.setChecked(not enabled)
        self.viewport.set_wireframe(enabled)

    def _take_path(self, path: str) -> None:
        suffix = Path(path).suffix.lower()
        if suffix == ".stl":
            self._load_stl(path)
        elif suffix in IMAGE_TYPES:
            self._load_image(path)

    def _short(self, text: str, limit: int = 22) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self._take_path(path)
        event.acceptProposedAction()

    def _fit_spin(self, spin: QDoubleSpinBox) -> None:
        spin.ensurePolished()
        width = max(spin.minimumSizeHint().width(), spin.sizeHint().width())
        if width > 0:
            spin.setFixedWidth(width)

    def _lock_panel_minimum(self) -> None:
        if not hasattr(self, "_panel"):
            return
        self._panel.ensurePolished()
        self._body.ensurePolished()
        margins = self._form.contentsMargins()
        content = self._body.minimumSizeHint().width()
        content = max(
            content,
            self.centered.sizeHint().width() + margins.left() + margins.right(),
            self.invert.sizeHint().width() + margins.left() + margins.right(),
            self._title.sizeHint().width() + margins.left() + margins.right(),
        )
        for spin in (
            self.depth_spin,
            self.scale_spin,
            self.contrast_spin,
            self.angle_spin,
            self.shift_x_spin,
            self.shift_y_spin,
        ):
            # Keep a usable groove beside the numeric field.
            row = spin.width() + 96 + 8
            content = max(content, row + margins.left() + margins.right())
        panel_margins = self._panel.layout().contentsMargins()
        # Border, vertical scrollbar, and a few pixels so glyphs clear the edge.
        chrome = panel_margins.left() + panel_margins.right() + 2 + 12 + 12
        floor = content + chrome
        if floor > self._panel.minimumWidth():
            self._panel.setMinimumWidth(floor)
        splitter = getattr(self, "_splitter", None)
        if not self._split_applied or splitter is None or splitter.width() <= 0:
            return
        sizes = splitter.sizes()
        if not sizes or sizes[0] + 1 >= self._panel.minimumWidth():
            return
        total = sum(sizes)
        right_floor = max(self._stage.minimumWidth(), 1)
        left = min(self._panel.minimumWidth(), max(1, total - right_floor))
        splitter.setSizes([left, max(1, total - left)])

    def _apply_initial_split(self) -> None:
        if self._split_applied:
            return
        total = self._splitter.width()
        if total < 200:
            return
        right_floor = max(self._stage.minimumWidth(), self.viewport.minimumWidth())
        left = max(self._panel.minimumWidth(), int(total * 0.54))
        left = min(left, total - right_floor)
        left = max(left, min(self._panel.minimumWidth(), total - right_floor))
        self._splitter.setSizes([left, max(1, total - left)])
        self._split_applied = True

    def _finish_layout(self) -> None:
        for spin in (
            self.depth_spin,
            self.scale_spin,
            self.contrast_spin,
            self.angle_spin,
            self.shift_x_spin,
            self.shift_y_spin,
        ):
            self._fit_spin(spin)
        for button in (self.frame_button, self.shaded_button, self.wire_button):
            button.setMinimumWidth(button.sizeHint().width())
        gaps = max(self._toolbar.count() - 1, 0) * self._toolbar.spacing()
        toolbar_floor = self.view_title.minimumWidth() + sum(
            button.minimumWidth() for button in (self.frame_button, self.shaded_button, self.wire_button)
        )
        self._stage.setMinimumWidth(max(toolbar_floor + gaps, self.viewport.minimumWidth()))
        self._lock_panel_minimum()
        self._apply_initial_split()
        self._lock_panel_minimum()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._finish_layout()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(1500)
        self.viewport.close()
        super().closeEvent(event)
