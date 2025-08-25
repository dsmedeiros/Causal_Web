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
        Text { text: "Results"; color: "white" }
        Row {
            spacing: 4
            Text { text: "Optimizer"; color: "white" }
            ComboBox {
                id: optBox
                model: ["", "mcts_h"]
                currentIndex: 1
                onActivated: resultsModel.optimizer = optBox.currentText
            }
        }
        Row {
            spacing: 4
            Text { text: "Promotion"; color: "white" }
            Slider {
                id: promoSlider
                width: 120
                from: 0
                to: 1
                value: resultsModel.promotionMin
                onValueChanged: resultsModel.promotionMin = value
            }
        }
        Row {
            spacing: 4
            Text { text: "Corr"; color: "white" }
            Slider {
                id: corrSlider
                width: 120
                from: 0
                to: 1
                value: resultsModel.proxyFullCorrMin
                onValueChanged: resultsModel.proxyFullCorrMin = value
            }
        }
        Button { text: "Refresh"; onClicked: resultsModel.refresh() }
        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: resultsModel.rows
            delegate: Rectangle {
                width: parent.width
                height: 24
                color: index % 2 === 0 ? "#202020" : "#303030"
                Text {
                    anchors.verticalCenter: parent.verticalCenter
                    color: "white"
                    text: model.run_id + " | " + model.promotion_rate.toFixed(2) + " | " + model.residual.toFixed(3)
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: resultsModel.openReplay(model.run_id)
                }
            }
        }
        Component.onCompleted: resultsModel.refresh()
    }
}
