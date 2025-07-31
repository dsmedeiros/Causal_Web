"""Display results of Bell inequality analysis."""

from __future__ import annotations

from typing import Dict, Tuple

try:
    from PySide6.QtCharts import (
        QChart,
        QChartView,
        QBarSet,
        QBarSeries,
        QBarCategoryAxis,
    )
    from PySide6.QtGui import QAction
except Exception:  # pragma: no cover - optional dependency
    QChart = object  # type: ignore
    QChartView = object  # type: ignore
    QBarSet = object  # type: ignore
    QBarSeries = object  # type: ignore
    QBarCategoryAxis = object  # type: ignore

from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel

from ..analysis.bell import compute_bell_statistics


class BellAnalysisWindow(QMainWindow):
    """Window presenting CHSH results and expectation histogram."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bell Inequality Analysis")
        self.resize(400, 300)

        central = QWidget()
        layout = QVBoxLayout(central)
        self.result_label = QLabel()
        layout.addWidget(self.result_label)
        self.setCentralWidget(central)

        self.chart_view: QChartView | None = None
        self._run_analysis()

    # ------------------------------------------------------------------
    def _run_analysis(self) -> None:
        """Compute statistics and update the display."""
        s_value, expectations = compute_bell_statistics()
        self.result_label.setText(f"S = {s_value:.3f}")
        if QChartView is object:
            return

        chart = QChart()
        bar_set = QBarSet("E")
        keys = ["A1B1", "A1B2", "A2B1", "A2B2"]
        for key in keys:
            bar_set << expectations.get((key[:2], key[2:]), 0.0)
        series = QBarSeries()
        series.append(bar_set)
        chart.addSeries(series)
        axis = QBarCategoryAxis()
        axis.append(keys)
        chart.createDefaultAxes()
        chart.setAxisX(axis, series)
        if self.chart_view is None:
            self.chart_view = QChartView(chart)
            self.centralWidget().layout().addWidget(self.chart_view)
        else:
            self.chart_view.setChart(chart)
