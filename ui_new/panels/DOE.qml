import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    property var panels
    property int replayIndex: -1
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4

        Repeater {
            model: doeModel.groups
            delegate: Row {
                spacing: 4
                Text { text: modelData.name; color: "white" }
                TextField {
                    id: lowField
                    width: 40
                    text: modelData.low
                    onEditingFinished: doeModel.setGroupRange(modelData.name, parseFloat(text), parseFloat(highField.text))
                }
                TextField {
                    id: highField
                    width: 40
                    text: modelData.high
                    onEditingFinished: doeModel.setGroupRange(modelData.name, parseFloat(lowField.text), parseFloat(text))
                }
                TextField {
                    id: stepField
                    width: 30
                    text: modelData.steps
                    onEditingFinished: doeModel.setGroupSteps(modelData.name, parseInt(text))
                }
            }
        }

        Row {
            spacing: 4
            ComboBox { id: modeBox; model: ["LHS", "Grid"] }
            TextField { id: sampleField; width: 40; text: "20" }
            Button {
                text: "Start"
                onClicked: {
                    if (modeBox.currentText === "LHS")
                        doeModel.runLhs(parseInt(sampleField.text))
                    else
                        doeModel.runGrid()
                }
            }
            Button { text: "Stop"; onClicked: doeModel.stop() }
            Button { text: "Resume"; onClicked: doeModel.resume() }
            Button { text: "Promote"; onClicked: doeModel.promote() }
        }

        ProgressBar { width: parent.width; value: doeModel.progress }
        Text { text: "ETA: " + doeModel.eta.toFixed(1) + "s"; color: "white" }

        Text { text: "Top-K"; color: "white" }
        ListView {
            width: parent.width
            height: 100
            model: doeModel.topK
            delegate: Item {
                width: parent.width
                height: 24
                Row {
                    spacing: 4
                    Text { text: (index + 1) + ":"; color: "white" }
                    Text { text: fitness.toFixed(3); color: "white" }
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        replayModel.load("experiments/" + path)
                        if (panels && replayIndex >= 0)
                            panels.currentIndex = replayIndex
                        }
                    }
                }
        }
        Text { text: "Scatter"; color: "white" }
        Canvas {
            id: scatter
            width: parent.width
            height: 150
            onPaint: {
                var ctx = getContext('2d')
                ctx.fillStyle = '#202020'
                ctx.fillRect(0,0,width,height)
                var pts = doeModel.scatter
                if (!pts) return
                ctx.fillStyle = 'red'
                for (var i=0;i<pts.length;i++) {
                    var p = pts[i]
                    var x = p[0] * width
                    var y = height - p[1] * height
                    ctx.fillRect(x, y, 3, 3)
                }
            }
            Connections { target: doeModel; function onScatterChanged() { scatter.requestPaint() } }
        }
        Text { text: "Parallel Coordinates"; color: "white" }
        Canvas {
            id: parallel
            width: parent.width
            height: 150
            onPaint: {
                var ctx = getContext('2d')
                ctx.fillStyle = '#202020'
                ctx.fillRect(0,0,width,height)
                var data = doeModel.parallel
                var names = doeModel.groupNames
                if (!data || data.length === 0) return
                var brushes = doeModel.brushes
                for (var i=0;i<data.length;i++) {
                    var run = data[i]
                    var match = true
                    for (var ax in brushes) {
                        var r = brushes[ax]
                        var val = run[ax]
                        if (val < r[0] || val > r[1]) { match = false; break }
                    }
                    ctx.strokeStyle = match ? 'rgba(0,255,255,0.7)' : 'rgba(0,255,255,0.1)'
                    ctx.beginPath()
                    for (var j=0;j<names.length;j++) {
                        var x = j/(names.length-1) * width
                        var y = height - run[j]*height
                        if (j===0) ctx.moveTo(x,y); else ctx.lineTo(x,y)
                    }
                    ctx.stroke()
                }
                for (var ax in brushes) {
                    var r = brushes[ax]
                    var x = ax/(names.length-1) * width - 5
                    var y1 = height - r[1]*height
                    var y2 = height - r[0]*height
                    ctx.fillStyle = 'rgba(255,255,0,0.2)'
                    ctx.fillRect(x, y1, 10, y2 - y1)
                }
            }
            Connections { target: doeModel; function onParallelChanged() { parallel.requestPaint() } }
            Connections { target: doeModel; function onBrushesChanged() { parallel.requestPaint() } }
            Connections { target: doeModel; function onGroupNamesChanged() { parallel.requestPaint() } }
            MouseArea {
                anchors.fill: parent
                property real startY: 0
                property int axis: 0
                onPressed: {
                    axis = Math.round(mouse.x / width * (doeModel.groupNames.length - 1))
                    startY = mouse.y
                }
                onReleased: {
                    var dy = Math.abs(mouse.y - startY)
                    var norm1 = 1 - Math.max(mouse.y, startY) / height
                    var norm2 = 1 - Math.min(mouse.y, startY) / height
                    if (dy < 5)
                        doeModel.setBrush(axis, 0, 0)
                    else
                        doeModel.setBrush(axis, norm1, norm2)
                }
            }
        }
        Text { text: "Heatmap"; color: "white" }
        Canvas {
            id: heatmap
            width: parent.width
            height: 150
            onPaint: {
                var ctx = getContext('2d')
                ctx.fillStyle = '#202020'
                ctx.fillRect(0,0,width,height)
                var grid = doeModel.heatmap
                if (!grid || grid.length === 0) return
                var rows = grid.length
                var cols = grid[0].length
                var minv = Infinity
                var maxv = -Infinity
                for (var y=0;y<rows;y++){
                    for (var x=0;x<cols;x++){
                        var v = grid[y][x]
                        if (v===null) continue
                        if (v<minv) minv=v
                        if (v>maxv) maxv=v
                    }
                }
                var cw = width/cols
                var ch = height/rows
                for (var y=0;y<rows;y++){
                    for (var x=0;x<cols;x++){
                        var val = grid[y][x]
                        if (val===null) continue
                        var norm = (val-minv)/((maxv-minv)||1)
                        ctx.fillStyle = 'rgba(255,0,0,'+norm+')'
                        ctx.fillRect(x*cw, height-(y+1)*ch, cw, ch)
                    }
                }
            }
            Connections { target: doeModel; function onHeatmapChanged() { heatmap.requestPaint() } }
        }
    }
}
