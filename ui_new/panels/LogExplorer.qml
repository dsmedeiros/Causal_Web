import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    color: "#202020"
    anchors.fill: parent

    ColumnLayout {
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
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: logsModel
            delegate: Text {
                text: modelData
                color: "white"
                visible: filterField.text === "" || modelData.toLowerCase().indexOf(filterField.text.toLowerCase()) !== -1
            }
        }
    }
}
