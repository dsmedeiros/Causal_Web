"""Service objects for building the main toolbar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from PySide6.QtGui import QAction
except ImportError:  # pragma: no cover - fallback for older PySide6
    from PySide6.QtWidgets import QAction
from PySide6.QtWidgets import QToolBar
from PySide6.QtCore import Qt

from .toolbar_builder import NodePanel, ConnectionPanel, ObserverPanel, MetaNodePanel


@dataclass
class ToolbarBuildService:
    """Construct the graph editing toolbar and associated panels."""

    main_window: Any

    # ------------------------------------------------------------------
    def build(self) -> QToolBar:
        toolbar = self._create_toolbar()
        self._create_panels()
        self._connect_signals(toolbar)
        return toolbar

    # ------------------------------------------------------------------
    def _create_toolbar(self) -> QToolBar:
        mw = self.main_window
        toolbar = QToolBar("Graph", mw)

        add_node_action = QAction("Add Node", mw)
        add_node_action.triggered.connect(mw.add_node)
        toolbar.addAction(add_node_action)

        add_conn_action = QAction("Add Connection", mw)
        add_conn_action.triggered.connect(mw.start_add_connection)
        toolbar.addAction(add_conn_action)
        mw.add_conn_action = add_conn_action

        add_obs_action = QAction("Add Observer", mw)
        add_obs_action.triggered.connect(mw.add_observer)
        toolbar.addAction(add_obs_action)

        add_meta_action = QAction("Add MetaNode", mw)
        add_meta_action.triggered.connect(mw.add_meta_node)
        toolbar.addAction(add_meta_action)

        layout_action = QAction("Auto Layout", mw)
        layout_action.triggered.connect(mw.canvas.auto_layout)
        toolbar.addAction(layout_action)

        return toolbar

    # ------------------------------------------------------------------
    def _create_panels(self) -> None:
        mw = self.main_window
        mw.node_panel = NodePanel(mw, mw.graph_window)
        mw.graph_window.addDockWidget(Qt.RightDockWidgetArea, mw.node_panel)
        mw.node_panel.hide()

        mw.connection_panel = ConnectionPanel(mw, mw.graph_window)
        mw.graph_window.addDockWidget(Qt.RightDockWidgetArea, mw.connection_panel)
        mw.connection_panel.hide()

        mw.observer_panel = ObserverPanel(mw, mw.graph_window)
        mw.graph_window.addDockWidget(Qt.RightDockWidgetArea, mw.observer_panel)
        mw.observer_panel.hide()

        mw.meta_node_panel = MetaNodePanel(mw, mw.graph_window)
        mw.graph_window.addDockWidget(Qt.RightDockWidgetArea, mw.meta_node_panel)
        mw.meta_node_panel.hide()

    # ------------------------------------------------------------------
    def _connect_signals(self, toolbar: QToolBar) -> None:
        mw = self.main_window
        mw.canvas.node_selected.connect(mw.node_panel.show_node)
        mw.canvas.connection_request.connect(mw.connection_panel.open_for)
        mw.canvas.connection_selected.connect(mw.connection_panel.show_connection)
        mw.canvas.meta_node_selected.connect(mw.meta_node_panel.show_meta_node)
        mw.canvas.observer_selected.connect(mw.observer_panel.open_for)
