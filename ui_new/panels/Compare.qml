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
            id: frameSlider
            from: 0
            to: compareModel.frameCount > 0 ? compareModel.frameCount - 1 : 0
            value: compareModel.frameIndex
            onMoved: compareModel.setFrame(Math.round(value))
            Connections { target: compareModel; function onFrameCountChanged(){ frameSlider.to = compareModel.frameCount > 0 ? compareModel.frameCount - 1 : 0 } }
        }
        Row {
            spacing: 4
            Button { text: "Prev"; onClicked: compareModel.prevFrame() }
            Button { text: timer.running ? "Pause" : "Play"; onClicked: timer.running ? timer.stop() : timer.start() }
            Button { text: "Next"; onClicked: compareModel.nextFrame() }
        }
        Timer {
            id: timer
            interval: 500
            repeat: true
            onTriggered: {
                compareModel.nextFrame()
                if (compareModel.frameIndex >= compareModel.frameCount - 1) timer.stop()
            }
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
