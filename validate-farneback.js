const outputEl = document.getElementById('output');
const runButton = document.getElementById('runButton');

let cvReady = false;
let webgpuReady = false;
let webgpuEngine = null;

function log(msg) {
    outputEl.textContent += `${msg}\n`;
}

function setReadyIfPossible() {
    if (cvReady && webgpuReady) {
        runButton.disabled = false;
        runButton.textContent = 'Run Validation';
    }
}

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

function bilinearSampleUint8(arr, width, height, x, y) {
    const cx = clamp(x, 0, width - 1);
    const cy = clamp(y, 0, height - 1);
    const x0 = Math.floor(cx);
    const y0 = Math.floor(cy);
    const x1 = Math.min(x0 + 1, width - 1);
    const y1 = Math.min(y0 + 1, height - 1);
    const tx = cx - x0;
    const ty = cy - y0;

    const i00 = y0 * width + x0;
    const i10 = y0 * width + x1;
    const i01 = y1 * width + x0;
    const i11 = y1 * width + x1;

    const v00 = arr[i00];
    const v10 = arr[i10];
    const v01 = arr[i01];
    const v11 = arr[i11];

    const vx0 = v00 + (v10 - v00) * tx;
    const vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * ty;
}

function createReference(width, height) {
    const img = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            const base =
                100 +
                50 * Math.sin(x * 0.09) +
                40 * Math.cos(y * 0.07) +
                30 * Math.sin((x + y) * 0.05);
            const checker = ((Math.floor(x / 16) + Math.floor(y / 16)) % 2) * 25;
            img[i] = clamp(Math.round(base + checker), 0, 255);
        }
    }
    return img;
}

function createTranslated(src, width, height, dx, dy) {
    const out = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // Flow is prev -> curr, so curr(x,y)=prev(x-dx,y-dy)
            out[y * width + x] = Math.round(bilinearSampleUint8(src, width, height, x - dx, y - dy));
        }
    }
    return out;
}

function toCvGrayMat(gray, width, height) {
    const rgba = new cv.Mat(height, width, cv.CV_8UC4);
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
        const v = gray[i];
        rgba.data[j] = v;
        rgba.data[j + 1] = v;
        rgba.data[j + 2] = v;
        rgba.data[j + 3] = 255;
    }
    const out = new cv.Mat();
    cv.cvtColor(rgba, out, cv.COLOR_RGBA2GRAY);
    rgba.delete();
    return out;
}

function computeErrorStats(flowArray, width, height, gtDx, gtDy, border = 24) {
    let absErrSum = 0;
    let rmsErrSum = 0;
    let maxErr = 0;
    let count = 0;
    let meanUx = 0;
    let meanUy = 0;
    const errors = [];

    for (let y = border; y < height - border; y++) {
        for (let x = border; x < width - border; x++) {
            const idx = (y * width + x) * 2;
            const ux = flowArray[idx];
            const uy = flowArray[idx + 1];
            const ex = ux - gtDx;
            const ey = uy - gtDy;
            const e = Math.hypot(ex, ey);

            absErrSum += e;
            rmsErrSum += e * e;
            maxErr = Math.max(maxErr, e);
            meanUx += ux;
            meanUy += uy;
            errors.push(e);
            count++;
        }
    }

    errors.sort((a, b) => a - b);
    const medianErr = errors[Math.floor(errors.length / 2)] || 0;
    const p90Err = errors[Math.floor(errors.length * 0.9)] || 0;
    const p99Err = errors[Math.floor(errors.length * 0.99)] || 0;

    return {
        meanU: meanUx / count,
        meanV: meanUy / count,
        meanAbsErr: absErrSum / count,
        rmse: Math.sqrt(rmsErrSum / count),
        medianErr,
        p90Err,
        p99Err,
        maxErr
    };
}

function flowFromCvMat(flowMat, width, height) {
    const out = new Float32Array(width * height * 2);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const base = y * width * 2 + x * 2;
            out[base] = flowMat.data32F[base];
            out[base + 1] = flowMat.data32F[base + 1];
        }
    }
    return out;
}

async function runValidation() {
    outputEl.textContent = '';
    log('Running Farneback cross-validation...');

    const width = 320;
    const height = 240;
    const tests = [
        { dx: 1.0, dy: 0.0 },
        { dx: 2.5, dy: -1.5 },
        { dx: -3.0, dy: 2.0 },
        { dx: 4.0, dy: 3.0 }
    ];

    const fbParams = {
        pyrScale: 0.5,
        levels: 3,
        winSize: 21,
        iterations: 3,
        polyN: 5,
        polySigma: 1.2,
        flags: 0
    };

    log(`Image size: ${width}x${height}`);
    log(`OpenCV Farneback params: ${JSON.stringify(fbParams)}`);
    log(`WebGPU engine version: ${WebGPUFarneback.VERSION || 'unknown'}`);
    log(`WebGPU params: ${JSON.stringify({
        maxLevels: webgpuEngine.maxLevels,
        minPyramidSize: webgpuEngine.minPyramidSize,
        iterationsPerLevel: webgpuEngine.iterationsPerLevel,
        updateBlendCoarse: webgpuEngine.updateBlendCoarse,
        updateBlendFine: webgpuEngine.updateBlendFine,
        smoothPassesCoarse: webgpuEngine.smoothPassesCoarse,
        smoothPassesFine: webgpuEngine.smoothPassesFine,
        finalSmoothPasses: webgpuEngine.finalSmoothPasses
    })}`);
    log('');

    for (const t of tests) {
        log(`Test displacement: dx=${t.dx.toFixed(2)}, dy=${t.dy.toFixed(2)}`);

        const prev = createReference(width, height);
        const curr = createTranslated(prev, width, height, t.dx, t.dy);

        const prevMat = toCvGrayMat(prev, width, height);
        const currMat = toCvGrayMat(curr, width, height);
        const cvFlow = new cv.Mat();

        cv.calcOpticalFlowFarneback(
            prevMat,
            currMat,
            cvFlow,
            fbParams.pyrScale,
            fbParams.levels,
            fbParams.winSize,
            fbParams.iterations,
            fbParams.polyN,
            fbParams.polySigma,
            fbParams.flags
        );

        const cvFlowArr = flowFromCvMat(cvFlow, width, height);
        const gpuFlowArr = await webgpuEngine.compute(prevMat.data, currMat.data, width, height, fbParams.winSize);

        const cvStats = computeErrorStats(cvFlowArr, width, height, t.dx, t.dy);
        const gpuStats = computeErrorStats(gpuFlowArr, width, height, t.dx, t.dy);

        let cvGpuDiff = 0;
        let cvGpuDiffRms = 0;
        let n = 0;
        const border = 24;
        for (let y = border; y < height - border; y++) {
            for (let x = border; x < width - border; x++) {
                const i = (y * width + x) * 2;
                const du = cvFlowArr[i] - gpuFlowArr[i];
                const dv = cvFlowArr[i + 1] - gpuFlowArr[i + 1];
                const d = Math.hypot(du, dv);
                cvGpuDiff += d;
                cvGpuDiffRms += d * d;
                n++;
            }
        }

        log(`  OpenCV mean(u,v)=(${cvStats.meanU.toFixed(3)}, ${cvStats.meanV.toFixed(3)})`);
        log(`  OpenCV MAE=${cvStats.meanAbsErr.toFixed(4)} Med=${cvStats.medianErr.toFixed(4)} P90=${cvStats.p90Err.toFixed(4)} P99=${cvStats.p99Err.toFixed(4)} RMSE=${cvStats.rmse.toFixed(4)} Max=${cvStats.maxErr.toFixed(4)}`);
        log(`  WebGPU mean(u,v)=(${gpuStats.meanU.toFixed(3)}, ${gpuStats.meanV.toFixed(3)})`);
        log(`  WebGPU MAE=${gpuStats.meanAbsErr.toFixed(4)} Med=${gpuStats.medianErr.toFixed(4)} P90=${gpuStats.p90Err.toFixed(4)} P99=${gpuStats.p99Err.toFixed(4)} RMSE=${gpuStats.rmse.toFixed(4)} Max=${gpuStats.maxErr.toFixed(4)}`);
        log(`  OpenCV vs WebGPU mean diff=${(cvGpuDiff / n).toFixed(4)} RMS diff=${Math.sqrt(cvGpuDiffRms / n).toFixed(4)}`);
        log('');

        prevMat.delete();
        currMat.delete();
        cvFlow.delete();
    }

    log('Done.');
}

runButton.addEventListener('click', () => {
    runValidation().catch(err => {
        log(`Validation failed: ${err.message}`);
        console.error(err);
    });
});

window.onOpenCvReady = function onOpenCvReady() {
    cv.onRuntimeInitialized = () => {
        cvReady = true;
        log('OpenCV ready.');
        setReadyIfPossible();
    };
};

(async () => {
    try {
        webgpuEngine = await WebGPUFarneback.create();
        webgpuReady = true;
        log('WebGPU Farneback ready.');
    } catch (err) {
        log(`WebGPU unavailable: ${err.message}`);
    }
    setReadyIfPossible();
})();