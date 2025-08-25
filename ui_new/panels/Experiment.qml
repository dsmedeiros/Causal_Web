import QtQuick
import QtQuick.Controls

Rectangle {
    property var graphView
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
            Button { text: "Start"; onClicked: experimentModel.start() }
            Button { text: "Pause"; onClicked: experimentModel.pause() }
            Button { text: "Step"; onClicked: experimentModel.step() }
            Button { text: "Reset"; onClicked: experimentModel.reset() }
        }
        Row {
            spacing: 4
            Text { text: "Rate"; color: "white" }
            Slider {
                width: 120
                from: 0.1
                to: 2.0
                value: experimentModel.rate
                onValueChanged: experimentModel.setRate(value)
            }
        }
        Row {
            spacing: 4
            CheckBox {
                text: "Labels"
                checked: graphView ? graphView.labelsVisible : true
                onToggled: if (graphView) graphView.labelsVisible = checked
            }
            CheckBox {
                text: "Edges"
                checked: graphView ? graphView.edgesVisible : true
                onToggled: if (graphView) graphView.edgesVisible = checked
            }
        }
    }
}
