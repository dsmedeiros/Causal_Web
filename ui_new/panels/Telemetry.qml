import QtQuick 2.15
import "."

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
        Text { text: "events/sec"; color: "white" }
        RollingPlot {
            width: parent.width
            height: 40
            points: telemetryModel.counters["events_per_sec"] || []
            band: telemetryModel.counterIntervals["events_per_sec"]
                  ? telemetryModel.counterIntervals["events_per_sec"].slice(1, 3)
                  : null
        }
        Text { text: "residual"; color: "white" }
        RollingPlot {
            width: parent.width
            height: 40
            points: telemetryModel.invariants["inv_conservation_residual"] || []
            band: telemetryModel.counterIntervals["inv_conservation_residual"]
                  ? telemetryModel.counterIntervals["inv_conservation_residual"].slice(1, 3)
                  : null
        }
    }
}
