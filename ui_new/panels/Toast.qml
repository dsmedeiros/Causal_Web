import QtQuick 2.15
import QtQuick.Controls 2.15

Popup {
    id: root
    property alias text: label.text
    x: (parent ? parent.width : 0) / 2 - width / 2
    y: parent ? parent.height - height - 20 : 0
    contentItem: Label {
        id: label
        color: "white"
        padding: 8
        background: Rectangle {
            color: "#323232"
            radius: 4
        }
    }
    Timer {
        id: hideTimer
        interval: 3000
        running: false
        repeat: false
        onTriggered: root.close()
    }
    function show(msg) {
        label.text = msg
        open()
        hideTimer.restart()
    }
}
