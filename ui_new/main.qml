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

    TabView {
        id: panels
        width: 250
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.right: parent.right

        Tab { title: "Telemetry"; Telemetry { anchors.fill: parent; graphView: graphView } }
        Tab { title: "Meters"; Meters { anchors.fill: parent } }
        Tab { title: "Experiment"; Experiment { anchors.fill: parent } }
        Tab { title: "Replay"; Replay { anchors.fill: parent } }
        Tab { title: "Logs"; LogExplorer { anchors.fill: parent } }
    }
}
