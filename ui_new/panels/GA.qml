import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    property var panels
    property int replayIndex: -1
    color: "#202020"
    anchors.fill: parent

    Connections {
        target: gaModel
        function onBaselinePromoted(path) {
            experimentModel.status = "Baseline saved to " + path
        }
    }

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4

        Grid {
            columns: 2
            rowSpacing: 4
            columnSpacing: 4
            Text { text: "Pop"; color: "white" }
            TextField { text: gaModel.populationSize; width: 40; onEditingFinished: gaModel.populationSize = parseInt(text) }
            Text { text: "Mut"; color: "white" }
            TextField { text: gaModel.mutationRate; width: 40; onEditingFinished: gaModel.mutationRate = parseFloat(text) }
            Text { text: "Cross"; color: "white" }
            TextField { text: gaModel.crossoverRate; width: 40; onEditingFinished: gaModel.crossoverRate = parseFloat(text) }
            Text { text: "Elite"; color: "white" }
            TextField { text: gaModel.elitism; width: 40; onEditingFinished: gaModel.elitism = parseInt(text) }
            Text { text: "Gen"; color: "white" }
            TextField { text: gaModel.maxGenerations; width: 40; onEditingFinished: gaModel.maxGenerations = parseInt(text) }
        }
        Row {
            spacing: 4
            Button {
                text: gaModel.running ? "Pause" : "Start"
                onClicked: gaModel.running ? gaModel.pause() : gaModel.start()
            }
            Button { text: "Resume"; onClicked: gaModel.resume() }
            Button { text: "Promote Baseline"; onClicked: gaModel.promoteBaseline() }
            Button { text: "Export Best"; onClicked: gaModel.exportBest() }
        }
        ListView {
            width: parent.width
            height: 100
            model: gaModel.population
            delegate: Item {
                width: parent.width
                height: 24
                Row {
                    spacing: 4
                    Text { text: fitness.toFixed(3); color: "white" }
                    Text { text: obj0.toFixed(2); color: "white" }
                    Text { text: obj1.toFixed(2); color: "white" }
                    Text { text: invCausality ? "C✓" : "C✗"; color: invCausality ? "lime" : "red" }
                    Text { text: invAncestry ? "A✓" : "A✗"; color: invAncestry ? "lime" : "red" }
                    Text { text: invResidual ? "R✓" : "R✗"; color: invResidual ? "lime" : "red" }
                    Text { text: invNoSignal ? "N✓" : "N✗"; color: invNoSignal ? "lime" : "red" }
                    Button {
                        text: "Replay"
                        enabled: path
                        onClicked: {
                            replayModel.load("experiments/" + path)
                            Qt.callLater(replayModel.play)
                            if (panels && replayIndex >= 0)
                                panels.currentIndex = replayIndex
                        }
                    }
                }
            }
        }
        Canvas {
            id: fitnessPlot
            width: parent.width
            height: 120
            onPaint: {
                var ctx = getContext('2d')
                ctx.fillStyle = '#202020'
                ctx.fillRect(0,0,width,height)
                var hist = gaModel.history
                if (!hist || hist.length < 2) return
                var minVal = Math.min.apply(null, hist)
                var maxVal = Math.max.apply(null, hist)
                ctx.strokeStyle = 'yellow'
                ctx.beginPath()
                for (var i=0;i<hist.length;i++) {
                    var x = i/(hist.length-1) * width
                    var y = height - (hist[i]-minVal)/((maxVal-minVal)||1) * height
                    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y)
                }
                ctx.stroke()
            }
            Connections { target: gaModel; function onHistoryChanged() { fitnessPlot.requestPaint() } }
        }
        Text { text: "Pareto Front"; color: "white" }
        Canvas {
            id: pareto
            width: parent.width
            height: 120
            onPaint: {
                var ctx = getContext('2d')
                ctx.fillStyle = '#202020'
                ctx.fillRect(0,0,width,height)
                var pts = gaModel.pareto
                if (!pts || pts.length === 0) return
                var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
                for (var i=0;i<pts.length;i++){
                    var p = pts[i]
                    if (p[0]<minX) minX=p[0]
                    if (p[0]>maxX) maxX=p[0]
                    if (p[1]<minY) minY=p[1]
                    if (p[1]>maxY) maxY=p[1]
                }
                ctx.fillStyle = 'cyan'
                for (var i=0;i<pts.length;i++){
                    var p = pts[i]
                    var x = (p[0]-minX)/((maxX-minX)||1) * width
                    var y = height - (p[1]-minY)/((maxY-minY)||1) * height
                    ctx.fillRect(x-2,y-2,4,4)
                }
            }
            Connections { target: gaModel; function onParetoChanged() { pareto.requestPaint() } }
        }
        ListView {
            width: parent.width
            height: 80
            model: gaModel.pareto
            delegate: Row {
                spacing: 4
                Text { text: Number(modelData[0]).toFixed(3); color: "white" }
                Text { text: Number(modelData[1]).toFixed(3); color: "white" }
                Button { text: "Promote"; onClicked: gaModel.promoteIndex(index) }
            }
        }
        Text { text: "Hall of Fame"; color: "white" }
        ListView {
            width: parent.width
            height: 80
            model: gaModel.hallOfFame
            delegate: Item {
                width: parent.width
                height: 24
                Row {
                    spacing: 4
                    Text { text: (typeof gen !== "undefined" ? gen : "?") + ":"; color: "white" }
                    Text { text: fitness.toFixed(3); color: "white" }
                    Button {
                        text: "Replay"
                        onClicked: {
                            replayModel.load("experiments/" + path)
                            Qt.callLater(replayModel.play)
                            if (panels && replayIndex >= 0)
                                panels.currentIndex = replayIndex
                        }
                    }
                    Button {
                        id: promoteBtn
                        text: "Promote"
                        onClicked: gaModel.promote(modelData)
                    }
                }
            }
        }
    }
}
