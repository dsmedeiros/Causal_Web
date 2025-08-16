import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4
        Text { text: "Replay"; color: "white" }
        Row {
            spacing: 4
            Button {
                text: "Play"
                onClicked: replayModel.play()
            }
            Button {
                text: "Pause"
                onClicked: replayModel.pause()
            }
        }
        Slider {
            from: 0
            to: 1
            value: replayModel.progress
            onMoved: replayModel.seek(value)
        }
        Text {
            text: Math.round(replayModel.progress * 100) + "%"
            color: "white"
        }
    }
}
