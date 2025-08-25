import QtQuick
import QtQuick.Controls

Rectangle {
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4
        Text { text: "Replay"; color: "white" }
        Row {
            spacing: 4
            Button {
                text: "Play"
                onClicked: replayModel.play()
            }
            Button {
                text: "Pause"
                onClicked: replayModel.pause()
            }
        }
        Row {
            spacing: 4
            TextField { id: logPath; placeholderText: "delta log" }
            Button { text: "Load"; onClicked: replayModel.load(logPath.text) }
        }
        Slider {
            from: 0
            to: 1
            value: replayModel.progress
            onMoved: replayModel.seek(value)
        }
        Text {
            text: Math.round(replayModel.progress * 100) + "%"
            color: "white"
        }
        Row {
            spacing: 4
            TextField { id: bookmarkName; placeholderText: "Bookmark" }
            Button { text: "Add"; onClicked: replayModel.addBookmark(bookmarkName.text) }
        }
        Row {
            spacing: 4
            TextField { id: annotationText; placeholderText: "Annotation" }
            Button { text: "Add"; onClicked: replayModel.addAnnotation(annotationText.text) }
        }
        ListView {
            height: 60
            model: replayModel.bookmarks
            delegate: Text {
                color: "white"
                text: model.name + " @ " + Math.round(model.progress * 100) + "%"
            }
        }
        ListView {
            height: 60
            model: replayModel.annotations
            delegate: Text {
                color: "white"
                text: Math.round(model.progress * 100) + "% - " + model.text
            }
        }
    }
}
