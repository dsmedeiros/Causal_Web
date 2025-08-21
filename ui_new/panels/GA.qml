import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: root
    property var panels
    property int replayIndex: -1
    property int xObj: 0
    property int yObj: 1
    property int compareIndex: -1
    property var compareSelection: []
    color: "#202020"
    anchors.fill: parent

    Connections {
        target: gaModel
        function onBaselinePromoted(path) {
            experimentModel.status = "Baseline saved to " + path
            toast.show("Baseline saved to " + path)
        }
    }

    Toast { id: toast }

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
        CheckBox {
            text: "Multi-objective"
            checked: gaModel.multiObjective
            onToggled: gaModel.multiObjective = checked
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
            Button { text: "Run baseline"; onClicked: experimentModel.runBaseline() }
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
                    CheckBox {
                        enabled: path !== ""
                        checked: root.compareSelection.indexOf("experiments/" + path) !== -1
                        onToggled: {
                            var p = "experiments/" + path
                            var idx = root.compareSelection.indexOf(p)
                            if (checked) {
                                if (idx === -1 && root.compareSelection.length < 2)
                                    root.compareSelection.push(p)
                                else
                                    checked = false
                            } else if (idx >= 0) {
                                root.compareSelection.splice(idx, 1)
                            }
                        }
                    }
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
        Button {
            text: "Compare"
            enabled: root.compareSelection.length === 2
            onClicked: {
                compareModel.loadRuns(root.compareSelection[0], root.compareSelection[1])
                if (panels && compareIndex >= 0)
                    panels.currentIndex = compareIndex
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
        Row {
            spacing: 4
            Text { text: "X"; color: "white" }
            ComboBox {
                model: gaModel.objectiveNames
                currentIndex: xObj
                onCurrentIndexChanged: xObj = currentIndex
                width: 120
            }
            Text { text: "Y"; color: "white" }
            ComboBox {
                model: gaModel.objectiveNames
                currentIndex: yObj
                onCurrentIndexChanged: yObj = currentIndex
                width: 120
            }
        }
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
                    var p = pts[i].objs
                    if (p[xObj]<minX) minX=p[xObj]
                    if (p[xObj]>maxX) maxX=p[xObj]
                    if (p[yObj]<minY) minY=p[yObj]
                    if (p[yObj]>maxY) maxY=p[yObj]
                }
                ctx.fillStyle = 'cyan'
                for (var i=0;i<pts.length;i++){
                    var p = pts[i].objs
                    var x = (p[xObj]-minX)/((maxX-minX)||1) * width
                    var y = height - (p[yObj]-minY)/((maxY-minY)||1) * height
                    ctx.fillRect(x-2,y-2,4,4)
                }
                ctx.fillStyle = 'white'
                var xLabel = gaModel.objectiveNames[xObj] || ''
                var yLabel = gaModel.objectiveNames[yObj] || ''
                ctx.fillText(xLabel, 4, 12)
                ctx.save()
                ctx.translate(0, height)
                ctx.rotate(-Math.PI/2)
                ctx.fillText(yLabel, 4, 12)
                ctx.restore()
            }
            Connections { target: gaModel; function onParetoChanged() { pareto.requestPaint() } }
        }
        Row {
            spacing: 4
            Text { text: "Rank"; color: "white" }
            Text { text: "Crowd"; color: "white" }
            Repeater {
                model: gaModel.objectiveNames
                delegate: Text { text: modelData; color: "white" }
            }
        }
        ListView {
            width: parent.width
            height: 80
            model: gaModel.pareto
            delegate: Row {
                id: rowItem
                property var rowData: modelData
                spacing: 4
                Text { text: rowData.rank; color: "white" }
                Text { text: rowData.crowding.toFixed(3); color: "white" }
                Repeater {
                    model: rowItem.rowData.objs.length
                    delegate: Text { text: Number(rowItem.rowData.objs[index]).toFixed(3); color: "white" }
                }
            }
        }
        Button { text: "Promote"; onClicked: paretoDialog.open(); enabled: gaModel.pareto.length > 0 }
        Dialog {
            id: paretoDialog
            modal: true
            standardButtons: Dialog.Close
            contentItem: ListView {
                width: 200
                height: 200
                model: gaModel.pareto
                delegate: Row {
                    id: dlgRow
                    property var rowData: modelData
                    spacing: 4
                    Text { text: index; color: "white" }
                    Repeater { model: dlgRow.rowData.objs.length; delegate: Text { text: Number(dlgRow.rowData.objs[index]).toFixed(2); color: "white" } }
                    Button { text: "Select"; onClicked: { gaModel.promoteIndex(index); paretoDialog.close() } }
                }
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
