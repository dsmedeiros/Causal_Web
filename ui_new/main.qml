import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import CausalGraph 1.0
import "panels"

Window {
    id: root
    property bool editMode: true
    property string tool: "select"
    property bool controlsEnabled: true
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
        enabled: root.controlsEnabled
    }

    // editing overlay
    MouseArea {
        id: editor
        anchors.fill: graphView
        enabled: root.editMode && root.controlsEnabled
        onPressed: function(mouse) {
            var pos = Qt.point(mouse.x, mouse.y)
            if (root.tool === "add") {
                store.add_node(pos.x, pos.y)
                var g = store.graph_arrays()
                graphView.set_graph(g.nodes, g.edges, g.labels, g.colors, g.flags)
            } else if (root.tool === "select") {
                var id = store.find_node(pos.x, pos.y)
                if (id >= 0) {
                    store.selectedNode = id
                } else {
                    var e = store.find_edge(pos.x, pos.y)
                    if (e >= 0) {
                        store.selectedEdge = e
                    } else {
                        var obs = store.find_observer(pos.x, pos.y)
                        if (obs >= 0) {
                            store.selectedObserver = obs
                        } else {
                            store.selectedBridge = store.find_bridge(pos.x, pos.y)
                        }
                    }
                }
            } else if (root.tool === "delete") {
                var id = store.find_node(pos.x, pos.y)
                store.delete_node(id)
                var g2 = store.graph_arrays()
                graphView.set_graph(g2.nodes, g2.edges, g2.labels, g2.colors, g2.flags)
            } else if (root.tool === "connect") {
                if (store.selectedNode < 0) {
                    store.selectedNode = store.find_node(pos.x, pos.y)
                } else {
                    var tgt = store.find_node(pos.x, pos.y)
                    store.connect_nodes(store.selectedNode, tgt)
                    store.selectedNode = -1
                    var g3 = store.graph_arrays()
                    graphView.set_graph(g3.nodes, g3.edges, g3.labels, g3.colors, g3.flags)
                }
            }
        }
        onPositionChanged: function(mouse) {
            if (root.tool === "select" && (mouse.buttons & Qt.LeftButton) && store.selectedNode >= 0) {
                store.move_node(store.selectedNode, mouse.x, mouse.y)
                var delta = {"node_positions": {}}
                delta.node_positions[store.selectedNode] = [mouse.x, mouse.y]
                graphView.apply_delta(delta)
            }
        }
    }

    Row {
        id: toolRow
        spacing: 4
        anchors.top: parent.top
        anchors.left: parent.left
        visible: root.editMode
        enabled: root.controlsEnabled
        Button { text: "Select"; onClicked: root.tool = "select" }
        Button { text: "Add"; onClicked: root.tool = "add" }
        Button { text: "Connect"; onClicked: root.tool = "connect" }
        Button { text: "Delete"; onClicked: root.tool = "delete" }
    }

    Button {
        id: modeButton
        text: root.editMode ? "Run" : "Edit"
        anchors.top: parent.top
        anchors.right: panels.left
        enabled: root.controlsEnabled
        onClicked: {
            root.editMode = !root.editMode
            graphView.editable = root.editMode
            if (!root.editMode) {
                store.load_graph()
            }
        }
    }

    Text {
        id: hud
        anchors.left: parent.left
        anchors.top: parent.top
        color: "white"
          text: "frame: " + metersModel.frame +
                " " + telemetryModel.depthLabel + ": " + telemetryModel.depth +
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
        enabled: root.controlsEnabled

        Tab { title: "Telemetry"; Telemetry { anchors.fill: parent; graphView: graphView } }
        Tab { title: "Meters"; Meters { anchors.fill: parent } }
        Tab { title: "Experiment"; Experiment { anchors.fill: parent; graphView: graphView } }
        Tab { id: replayTab; title: "Replay"; Replay { anchors.fill: parent } }
        Tab { title: "Logs"; LogExplorer { anchors.fill: parent } }
        Tab { title: "Inspector"; Inspector { anchors.fill: parent } }
        Tab { title: "Validation"; Validation { anchors.fill: parent } }
        Tab { title: "DOE"; DOE { anchors.fill: parent; panels: panels; replayTab: replayTab } }
        Tab { title: "GA"; GA { anchors.fill: parent; panels: panels; replayTab: replayTab } }
        Tab { title: "Compare"; Compare { anchors.fill: parent } }
    }
}
