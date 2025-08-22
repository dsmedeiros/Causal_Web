import QtQuick 2.15

Canvas {
    id: canvas
    property var points: []
    onPaint: {
        var ctx = getContext('2d');
        ctx.reset();
        ctx.strokeStyle = 'lime';
        ctx.beginPath();
        var pts = points || [];
        if (pts.length > 0) {
            var max = Math.max.apply(Math, pts);
            if (max <= 0) max = 1;
            var step = width / Math.max(1, pts.length - 1);
            ctx.moveTo(0, height - (pts[0] / max) * height);
            for (var i = 1; i < pts.length; ++i)
                ctx.lineTo(i * step, height - (pts[i] / max) * height);
        }
        ctx.stroke();
    }
    onPointsChanged: requestPaint()
}
