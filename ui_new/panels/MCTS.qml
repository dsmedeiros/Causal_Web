import QtQuick 2.15
import QtQuick.Controls 2.15
import "."

Rectangle {
    id: root
    property var panels
    property int replayIndex: -1
    color: "#202020"
    anchors.fill: parent

    Connections {
        target: mctsModel
        function onBaselinePromoted(path) {
            experimentModel.status = "Baseline saved to " + path
            toast.show("Baseline saved to " + path)
        }
    }

    Toast { id: toast }

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4

        Grid {
            columns: 2
            rowSpacing: 4
            columnSpacing: 4
            Text { text: "c_ucb"; color: "white" }
            TextField { text: mctsModel.cUcb; width: 50; onEditingFinished: mctsModel.cUcb = parseFloat(text) }
            Text { text: "alpha_pw"; color: "white" }
            TextField { text: mctsModel.alphaPw; width: 50; onEditingFinished: mctsModel.alphaPw = parseFloat(text) }
            Text { text: "k_pw"; color: "white" }
            TextField { text: mctsModel.kPw; width: 50; onEditingFinished: mctsModel.kPw = parseFloat(text) }
            Text { text: "bins"; color: "white" }
            TextField { text: mctsModel.bins; width: 50; onEditingFinished: mctsModel.bins = parseInt(text) }
            Text { text: "promote"; color: "white" }
            TextField { text: mctsModel.promoteThreshold; width: 50; onEditingFinished: mctsModel.promoteThreshold = parseFloat(text) }
            Text { text: "promote_q"; color: "white" }
            TextField { text: mctsModel.promoteQuantile; width: 50; onEditingFinished: mctsModel.promoteQuantile = parseFloat(text) }
            Text { text: "promote_w"; color: "white" }
            TextField { text: mctsModel.promoteWindow; width: 50; onEditingFinished: mctsModel.promoteWindow = parseInt(text) }
            Text { text: "max_nodes"; color: "white" }
            TextField { text: mctsModel.maxNodes; width: 60; onEditingFinished: mctsModel.maxNodes = parseInt(text) }
            Text { text: "proxy_frames"; color: "white" }
            TextField { text: mctsModel.proxyFrames; width: 60; onEditingFinished: mctsModel.proxyFrames = parseInt(text) }
            Text { text: "full_frames"; color: "white" }
            TextField { text: mctsModel.fullFrames; width: 60; onEditingFinished: mctsModel.fullFrames = parseInt(text) }
        }
        CheckBox {
            text: "Multi-objective"
            checked: mctsModel.multiObjective
            onToggled: mctsModel.multiObjective = checked
        }
        Row {
            spacing: 4
            Button {
                text: mctsModel.running ? "Pause" : "Start"
                onClicked: mctsModel.running ? mctsModel.pause() : mctsModel.start()
            }
            Button { text: "Resume"; onClicked: mctsModel.resume() }
            Button { text: "Promote Baseline"; onClicked: mctsModel.promoteBaseline() }
            Button { text: "Local Ablation"; onClicked: mctsModel.localAblation() }
        }
        Row {
            spacing: 8
            Text { text: "Nodes: " + mctsModel.nodeCount; color: "white" }
            Text { text: "Frontier: " + mctsModel.frontier; color: "white" }
        }
        Row {
            spacing: 8
            Text { text: "Proxy: " + mctsModel.proxyEvaluations; color: "white" }
            Text { text: "Full: " + mctsModel.fullEvaluations; color: "white" }
        }
        Row {
            spacing: 4
            Text { text: "Expand"; color: "white" }
            ProgressBar { width: 80; value: mctsModel.expansionRate }
            Text { text: "Promote"; color: "white" }
            ProgressBar { width: 80; value: mctsModel.promotionRate }
        }
        Text { text: "Hall of Fame"; color: "white" }
        ListView {
            width: parent.width
            height: 100
            model: mctsModel.hallOfFame
            delegate: Item {
                width: parent.width
                height: 24
                Row {
                    spacing: 4
                    Text { text: fitness.toFixed(3); color: "white" }
                    Button {
                        text: "Replay"
                        enabled: path !== undefined && path !== ""
                        onClicked: {
                            replayModel.load("experiments/" + path)
                            Qt.callLater(replayModel.play)
                            if (panels && replayIndex >= 0)
                                panels.currentIndex = replayIndex
                        }
                    }
                }
            }
        }

        Text { text: "Local Ablations"; color: "white"; visible: mctsModel.ablations.length > 0 }
        Repeater {
            model: mctsModel.ablations
            delegate: Column {
                width: parent.width
                spacing: 4
                Text { text: modelData.params.join(", "); color: "white" }
                Item {
                    width: parent.width
                    height: 100
                    RollingPlot {
                        anchors.fill: parent
                        points: modelData.scores.map(function(row) { return row[0]; })
                        visible: modelData.params.length === 1
                    }
                    AblationHeatmap {
                        anchors.fill: parent
                        scores: modelData.scores
                        visible: modelData.params.length === 2
                    }
                }
            }
        }
    }
}
