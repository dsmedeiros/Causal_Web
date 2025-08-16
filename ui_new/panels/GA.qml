import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    color: "#202020"
    anchors.fill: parent

    Column {
        anchors.margins: 8
        anchors.fill: parent
        spacing: 4

        Row {
            spacing: 4
            Button { text: "Step"; onClicked: gaModel.step() }
            Button { text: "Promote"; onClicked: gaModel.promote() }
        }
        ListView {
            width: parent.width
            height: 100
            model: gaModel.population
            delegate: Row {
                spacing: 4
                Text { text: fitness.toFixed(3); color: "white" }
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
    }
}
