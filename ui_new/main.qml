import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import CausalGraph 1.0
import "panels"

Window {
    id: root
    width: 800
    height: 600
    visible: true
    title: qsTr("Causal Web - New UI")

    // QSGGeometry-backed GPU renderer
    GraphView {
        id: graphView
        objectName: "graphView"
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: panels.left
    }

    Text {
        id: hud
        anchors.left: parent.left
        anchors.top: parent.top
        color: "white"
        text: "frame: " + metersModel.frame +
              " depth: " + telemetryModel.depth +
              " windows: " +
              (telemetryModel.counters["window"] ? telemetryModel.counters["window"][telemetryModel.counters["window"].length - 1] : 0) +
              " bridges: " +
              (telemetryModel.counters["active_bridges"] ? telemetryModel.counters["active_bridges"][telemetryModel.counters["active_bridges"].length - 1] : 0) +
              " fps: " + metersModel.fps.toFixed(1) +
              " events/s: " +
              (telemetryModel.counters["events_per_sec"] ? telemetryModel.counters["events_per_sec"][telemetryModel.counters["events_per_sec"].length - 1].toFixed(1) : 0) +
              " residual: " + experimentModel.residual.toFixed(3)
        z: 10
    }

    TabView {
        id: panels
        width: 250
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.right: parent.right

        Tab { title: "Telemetry"; Telemetry { anchors.fill: parent; graphView: graphView } }
        Tab { title: "Meters"; Meters { anchors.fill: parent } }
        Tab { title: "Experiment"; Experiment { anchors.fill: parent; graphView: graphView } }
        Tab { title: "Replay"; Replay { anchors.fill: parent } }
        Tab { title: "Logs"; LogExplorer { anchors.fill: parent } }
    }
}
