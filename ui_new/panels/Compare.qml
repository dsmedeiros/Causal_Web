import QtQuick 2.15
import QtQuick.Controls 2.15
import Qt5Compat.GraphicalEffects 1.15

Rectangle {
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4
        Text { text: "Compare"; color: "white" }
        Row {
            spacing: 4
            TextField { id: runA; placeholderText: "Run A" }
            TextField { id: runB; placeholderText: "Run B" }
            Button {
                text: "Load"
                onClicked: compareModel.loadRuns(runA.text, runB.text)
            }
        }
        Slider {
            from: 0
            to: compareModel.frameCount > 0 ? compareModel.frameCount - 1 : 0
            onMoved: compareModel.setFrame(Math.round(value))
        }
        Row {
            spacing: 4
            Column {
                spacing: 2
                Text { text: "Run A"; color: "white" }
                Image {
                    id: imgA
                    source: compareModel.frameA
                    width: 200; height: 200
                    fillMode: Image.PreserveAspectFit
                }
            }
            Column {
                spacing: 2
                Text { text: "Run B"; color: "white" }
                Image {
                    id: imgB
                    source: compareModel.frameB
                    width: 200; height: 200
                    fillMode: Image.PreserveAspectFit
                }
            }
            Column {
                spacing: 2
                Text { text: "Diff"; color: "white" }
                Blend {
                    source: imgA
                    foregroundSource: imgB
                    mode: Blend.Difference
                    width: 200; height: 200
                }
            }
        }
        ListView {
            height: 100
            model: compareModel.metricDelta
            delegate: Text {
                color: "white"
                text: model.category + ": " + model.delta
            }
        }
    }
}
