import QtQuick 2.15

Canvas {
    id: canvas
    property var points: []
    property var band: null // [lower, upper]
    onPaint: {
        var ctx = getContext('2d');
        ctx.reset();
        var pts = points || [];
        var maxVal = 1;
        if (pts.length > 0)
            maxVal = Math.max(maxVal, Math.max.apply(Math, pts));
        if (band && band.length === 2) {
            maxVal = Math.max(maxVal, band[1]);
            var lowerY = height - (band[0] / maxVal) * height;
            var upperY = height - (band[1] / maxVal) * height;
            ctx.fillStyle = 'rgba(0,255,0,0.2)';
            ctx.fillRect(0, upperY, width, lowerY - upperY);
        }
        ctx.strokeStyle = 'lime';
        ctx.beginPath();
        if (pts.length > 0) {
            var step = width / Math.max(1, pts.length - 1);
            ctx.moveTo(0, height - (pts[0] / maxVal) * height);
            for (var i = 1; i < pts.length; ++i)
                ctx.lineTo(i * step, height - (pts[i] / maxVal) * height);
        }
        ctx.stroke();
    }
    onPointsChanged: requestPaint()
    onBandChanged: requestPaint()
}
