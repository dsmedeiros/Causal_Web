import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: root
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        spacing: 4
        Repeater {
            model: policyModel.actionNames
            delegate: Row {
                spacing: 4
                CheckBox {
                    checked: true
                    onToggled: policyModel.setActionEnabled(modelData, checked)
                }
                Text { text: modelData; color: "white" }
            }
        }
        Row {
            spacing: 4
            Text { text: "Horizon"; color: "white" }
            SpinBox {
                from: 2
                to: 5
                value: policyModel.horizon
                onValueChanged: policyModel.horizon = value
            }
        }
        Row {
            spacing: 4
            Button { text: "Plan"; onClicked: policyModel.plan() }
            Button { text: "Apply"; onClicked: policyModel.apply() }
        }
        Text { text: "Plan: " + policyModel.planSummary; color: "white" }
    }
}
