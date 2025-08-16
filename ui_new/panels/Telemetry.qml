import QtQuick 2.15

Rectangle {
    property var graphView
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4
        Text { text: "Telemetry"; color: "white" }
        Text {
            text: "nodes: " + telemetryModel.nodeCount
            color: "white"
        }
        Text {
            text: "edges: " + telemetryModel.edgeCount
            color: "white"
        }
        Text {
            text: "labels visible: " + (graphView.labelsVisible ? "yes" : "no")
            color: "white"
        }
        Text {
            text: "edges visible: " + (graphView.edgesVisible ? "yes" : "no")
            color: "white"
        }
    }
}
