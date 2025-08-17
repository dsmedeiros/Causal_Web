import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4

        Button { text: "Run LHS"; onClicked: doeModel.runLhs(20) }
        Text { text: "Top-K"; color: "white" }
        ListView {
            width: parent.width
            height: 100
            model: doeModel.topK
            delegate: Row {
                spacing: 4
                Text { text: (index + 1) + ":"; color: "white" }
                Text { text: fitness.toFixed(3); color: "white" }
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
                for (var i=0;i<data.length;i++) {
                    var run = data[i]
                    ctx.strokeStyle = 'rgba(0,255,255,0.3)'
                    ctx.beginPath()
                    for (var j=0;j<names.length;j++) {
                        var x = j/(names.length-1) * width
                        var y = height - run[j]*height
                        if (j===0) ctx.moveTo(x,y); else ctx.lineTo(x,y)
                    }
                    ctx.stroke()
                }
            }
            Connections { target: doeModel; function onParallelChanged() { parallel.requestPaint() } }
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
