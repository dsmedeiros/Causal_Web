import QtQuick 2.15

Rectangle {
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4
        Text { text: "Meters"; color: "white" }
        Text {
            text: "fps: " + metersModel.fps.toFixed(1)
            color: "white"
        }
    }
}
