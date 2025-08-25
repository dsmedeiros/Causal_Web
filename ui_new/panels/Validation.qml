import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    ColumnLayout {
        anchors.fill: parent
        spacing: 4
        Button {
            text: "Run Validation"
            onClicked: {
                warningsModel.clear()
                var results = store.validate_graph()
                for (var i = 0; i < results.length; ++i) {
                    warningsModel.append({ msg: results[i] })
                }
            }
        }
        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: ListModel { id: warningsModel }
            delegate: Text { text: msg }
        }
    }
}
