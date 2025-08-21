import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4
        Text { text: "Log Explorer"; color: "white" }
        Row {
            spacing: 4
            TextField {
                id: filterField
                placeholderText: "Filter"
            }
            Button {
                text: "Clear"
                onClicked: logsModel.clear()
            }
        }
        ListView {
            anchors.fill: parent
            model: logsModel
            delegate: Text {
                text: modelData
                color: "white"
                visible: filterField.text === "" || modelData.toLowerCase().indexOf(filterField.text.toLowerCase()) !== -1
            }
        }
    }
}
