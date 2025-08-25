import QtQuick

Canvas {
    id: canvas
    property var scores: []
    onPaint: {
        var ctx = getContext('2d');
        ctx.reset();
        var rows = scores.length;
        if (rows === 0)
            return;
        var cols = scores[0].length;
        var min = scores[0][0];
        var max = scores[0][0];
        for (var i = 0; i < rows; ++i) {
            for (var j = 0; j < cols; ++j) {
                var s = scores[i][j];
                if (s < min) min = s;
                if (s > max) max = s;
            }
        }
        var w = width / cols;
        var h = height / rows;
        for (var i = 0; i < rows; ++i) {
            for (var j = 0; j < cols; ++j) {
                var s = scores[i][j];
                var t = (s - min) / (max - min || 1);
                ctx.fillStyle = Qt.hsla(0.6 - 0.6 * t, 1, 0.5, 1);
                ctx.fillRect(j * w, (rows - 1 - i) * h, w, h);
            }
        }
    }
    onScoresChanged: requestPaint()
}
