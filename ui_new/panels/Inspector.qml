import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    Column {
        anchors.fill: parent
        spacing: 4
        Text {
            id: noneLabel
            text: "No selection"
            visible: store.selectedNode < 0 && store.selectedEdge < 0 && store.selectedObserver < 0 && store.selectedBridge < 0
        }
        // Node inspector
        Column {
            id: nodeCol
            spacing: 4
            visible: store.selectedNode >= 0
            Text { text: "Node " + store.selectedNode }
            TextField {
                id: nodeLabel
                text: store.selectedNode >= 0 ? store.get_node(store.selectedNode).label || "" : ""
                onEditingFinished: store.set_node_label(store.selectedNode, text)
            }
            Text { text: store.selectedNode >= 0 ? "x: " + store.get_node(store.selectedNode).x.toFixed(1) : "" }
            Text { text: store.selectedNode >= 0 ? "y: " + store.get_node(store.selectedNode).y.toFixed(1) : "" }
        }
        // Edge inspector
        Column {
            id: edgeCol
            spacing: 4
            visible: store.selectedEdge >= 0
            property var edge: store.get_edge(store.selectedEdge)
            Text { text: "Edge " + store.selectedEdge }
            Text { text: edge ? ("from: " + edge.from + " to: " + edge.to) : "" }
            TextField {
                id: edgeDelay
                text: edge && edge.delay !== undefined ? edge.delay : ""
                onEditingFinished: {
                    var val = parseInt(text);
                    if (!isNaN(val)) {
                        store.set_edge_delay(store.selectedEdge, val);
                    }
                }
            }
        }
        // Observer inspector
        Column {
            id: obsCol
            spacing: 4
            visible: store.selectedObserver >= 0
            property var obs: store.get_observer(store.selectedObserver)
            Text { text: obs ? ("Observer " + obs.id) : "" }
            TextField {
                id: obsId
                text: obs ? obs.id || "" : ""
                onEditingFinished: store.set_observer_id(store.selectedObserver, text)
            }
            TextField {
                id: obsFreq
                text: obs && obs.frequency !== undefined ? obs.frequency : ""
                onEditingFinished: {
                    var freq = parseFloat(text);
                    if (!isNaN(freq)) {
                        store.set_observer_frequency(store.selectedObserver, freq);
                    }
                }
            }
        }
        // Bridge inspector
        Column {
            id: bridgeCol
            spacing: 4
            visible: store.selectedBridge >= 0
            property var bridge: store.get_bridge(store.selectedBridge)
            Text { text: "Bridge " + store.selectedBridge }
            CheckBox {
                id: entangledBox
                text: "Entangled"
                checked: bridge && bridge.is_entangled ? true : false
                onToggled: store.set_bridge_entangled(store.selectedBridge, checked)
            }
        }
    }
    Connections {
        target: store
        onSelectionChanged: {
            nodeLabel.text = store.selectedNode >= 0 ? store.get_node(store.selectedNode).label || "" : ""
        }
        onEdgeSelectionChanged: {
            edgeCol.edge = store.get_edge(store.selectedEdge)
            edgeDelay.text = edgeCol.edge && edgeCol.edge.delay !== undefined ? edgeCol.edge.delay : ""
        }
        onObserverSelectionChanged: {
            obsCol.obs = store.get_observer(store.selectedObserver)
            obsId.text = obsCol.obs ? obsCol.obs.id || "" : ""
            obsFreq.text = obsCol.obs && obsCol.obs.frequency !== undefined ? obsCol.obs.frequency : ""
        }
        onBridgeSelectionChanged: {
            bridgeCol.bridge = store.get_bridge(store.selectedBridge)
            entangledBox.checked = bridgeCol.bridge && bridgeCol.bridge.is_entangled ? true : false
        }
    }
}
