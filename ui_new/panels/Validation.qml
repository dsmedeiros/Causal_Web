import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    Column {
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
            anchors.fill: parent
            model: ListModel { id: warningsModel }
            delegate: Text { text: msg }
        }
    }
}
