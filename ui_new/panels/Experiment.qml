import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4
        Text { text: "Experiment"; color: "white" }
        Text {
            text: "Status: " + experimentModel.status
            color: "white"
        }
        Text {
            text: "Residual: " + experimentModel.residual.toFixed(3)
            color: "white"
        }
        Row {
            spacing: 4
            Button {
                text: "Start"
                onClicked: experimentModel.start()
            }
            Button {
                text: "Pause"
                onClicked: experimentModel.pause()
            }
            Button {
                text: "Resume"
                onClicked: experimentModel.resume()
            }
            Button {
                text: "Reset"
                onClicked: experimentModel.reset()
            }
        }
    }
}
