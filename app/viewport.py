"""Embedded PyVista view used by the Finish window."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QSizePolicy
from pyvistaqt import QtInteractor


class FinishViewport(QtInteractor):
    file_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent, lighting="three lights", multi_samples=8)
        self.set_background("#101113", top="#1c1b19")
        self.add_axes(
            interactive=False,
            line_width=2,
            x_color="#c45c4a",
            y_color="#7d9a62",
            z_color="#6a8caf",
            viewport=(0.0, 0.0, 0.18, 0.18),
        )
        self.enable_anti_aliasing("fxaa")
        self._actor = None
        self._framed = False
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(QSize(180, 160))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.file_dropped.emit(path)
        event.acceptProposedAction()

    def show_mesh(self, vertices: np.ndarray, faces: np.ndarray, *, frame: bool) -> None:
        packed = np.hstack(
            (
                np.full((len(faces), 1), 3, dtype=np.int64),
                np.asarray(faces, dtype=np.int64),
            )
        ).ravel()
        poly = pv.PolyData(np.asarray(vertices, dtype=np.float64), packed)
        poly.compute_normals(
            inplace=True,
            feature_angle=50,
            split_vertices=True,
            auto_orient_normals=False,
        )
        if self._actor is not None:
            self.remove_actor(self._actor)
        self._actor = self.add_mesh(
            poly,
            color="#d5cfc3",
            smooth_shading=True,
            specular=0.28,
            specular_power=24,
            ambient=0.25,
            diffuse=0.82,
        )
        if frame or not self._framed:
            self.view_isometric()
            self.reset_camera()
            self.camera.zoom(1.15)
            self._framed = True
        self.render()

    def set_wireframe(self, enabled: bool) -> None:
        if self._actor is None:
            return
        prop = self._actor.GetProperty()
        if enabled:
            prop.SetRepresentationToWireframe()
            prop.SetColor(0.78, 0.75, 0.69)
            prop.SetLineWidth(1)
        else:
            prop.SetRepresentationToSurface()
            prop.SetColor(0.835, 0.812, 0.765)
        self.render()

    def frame_view(self) -> None:
        self.view_isometric()
        self.reset_camera()
        self.camera.zoom(1.15)
        self._framed = True
        self.render()
