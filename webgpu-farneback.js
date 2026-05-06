class WebGPUFarneback {
    static VERSION = '2026-05-06-c2f-v4';

    constructor(device) {
        this.device = device;
        this.pipeline = null;
        this.paramsBuffer = null;
        this.prevBuffer = null;
        this.currBuffer = null;
        this.flowBuffer = null;
        this.readBuffer = null;
        this.width = 0;
        this.height = 0;
        this.pixelCount = 0;

        // Coarse-to-fine settings
        this.maxLevels = 5;
        this.minPyramidSize = 24;
        this.iterationsPerLevel = 3;
        this.updateBlendCoarse = 0.45;
        this.updateBlendFine = 0.45;
        this.smoothPassesCoarse = 2;
        this.smoothPassesFine = 2;
        this.finalSmoothPasses = 0;
    }

    static async create() {
        if (!('gpu' in navigator)) {
            throw new Error('WebGPU is not supported in this browser');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('Failed to get WebGPU adapter');
        }

        const device = await adapter.requestDevice();
        const instance = new WebGPUFarneback(device);
        instance._initPipeline();
        return instance;
    }

    _initPipeline() {
        const shaderCode = `
struct Params {
    width: u32,
    height: u32,
    winRadius: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> prevGray: array<f32>;
@group(0) @binding(1) var<storage, read> currGray: array<f32>;
@group(0) @binding(2) var<storage, read_write> outFlow: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn clampCoord(v: i32, maxv: i32) -> i32 {
    return max(0, min(v, maxv));
}

fn idx(x: i32, y: i32, w: i32) -> i32 {
    return y * w + x;
}

fn I(arr: ptr<storage, array<f32>, read>, x: i32, y: i32, w: i32, h: i32) -> f32 {
    let xx = clampCoord(x, w - 1);
    let yy = clampCoord(y, h - 1);
    return (*arr)[idx(xx, yy, w)];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(params.width);
    let h = i32(params.height);

    if (i32(gid.x) >= w || i32(gid.y) >= h) {
        return;
    }

    let x = i32(gid.x);
    let y = i32(gid.y);
    let wr = i32(params.winRadius);
    let sigma = max(1.0, f32(wr) * 0.5);
    let inv2Sigma2 = 0.5 / (sigma * sigma);

    var sIxIx: f32 = 0.0;
    var sIxIy: f32 = 0.0;
    var sIyIy: f32 = 0.0;
    var sIxIt: f32 = 0.0;
    var sIyIt: f32 = 0.0;

    for (var dy: i32 = -wr; dy <= wr; dy = dy + 1) {
        for (var dx: i32 = -wr; dx <= wr; dx = dx + 1) {
            let xx = x + dx;
            let yy = y + dy;
            let r2 = f32(dx * dx + dy * dy);
            let wgt = exp(-r2 * inv2Sigma2);

            let ix = 0.5 * (I(&currGray, xx + 1, yy, w, h) - I(&currGray, xx - 1, yy, w, h));
            let iy = 0.5 * (I(&currGray, xx, yy + 1, w, h) - I(&currGray, xx, yy - 1, w, h));
            let it = I(&currGray, xx, yy, w, h) - I(&prevGray, xx, yy, w, h);

            sIxIx = sIxIx + wgt * ix * ix;
            sIxIy = sIxIy + wgt * ix * iy;
            sIyIy = sIyIy + wgt * iy * iy;
            sIxIt = sIxIt + wgt * ix * it;
            sIyIt = sIyIt + wgt * iy * it;
        }
    }

    let detRaw = sIxIx * sIyIy - sIxIy * sIxIy;
    let trace = sIxIx + sIyIy;
    var u: f32 = 0.0;
    var v: f32 = 0.0;

    // Reject updates in low-texture / near-singular regions to prevent large spikes.
    if (detRaw > 1e-5 && trace > 1e-3) {
        let det = detRaw + 1e-6;
        u = (sIxIy * sIyIt - sIyIy * sIxIt) / det;
        v = (sIxIy * sIxIt - sIxIx * sIyIt) / det;

        // Clamp per-iteration increment; global displacement is recovered across pyramid levels.
        let maxStep = max(1.0, f32(wr));
        u = clamp(u, -maxStep, maxStep);
        v = clamp(v, -maxStep, maxStep);
    }

    let outIndex = idx(x, y, w) * 2;
    outFlow[outIndex] = u;
    outFlow[outIndex + 1] = v;
}
`;

        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({ code: shaderCode }),
                entryPoint: 'main'
            }
        });

        this.paramsBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    _ensureBuffers(width, height) {
        const pixelCount = width * height;
        if (this.width === width && this.height === height && this.prevBuffer) {
            return;
        }

        this.width = width;
        this.height = height;
        this.pixelCount = pixelCount;

        const inputBytes = pixelCount * 4;
        const flowBytes = pixelCount * 2 * 4;

        if (this.prevBuffer) this.prevBuffer.destroy();
        if (this.currBuffer) this.currBuffer.destroy();
        if (this.flowBuffer) this.flowBuffer.destroy();
        if (this.readBuffer) this.readBuffer.destroy();

        this.prevBuffer = this.device.createBuffer({
            size: inputBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.currBuffer = this.device.createBuffer({
            size: inputBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.flowBuffer = this.device.createBuffer({
            size: flowBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        this.readBuffer = this.device.createBuffer({
            size: flowBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

    }

    _toOdd(v) {
        return v % 2 === 0 ? v + 1 : v;
    }

    _downsample2x(input, width, height) {
        const outW = Math.max(1, Math.floor(width / 2));
        const outH = Math.max(1, Math.floor(height / 2));
        const out = new Float32Array(outW * outH);

        for (let y = 0; y < outH; y++) {
            for (let x = 0; x < outW; x++) {
                const x0 = x * 2;
                const y0 = y * 2;
                const x1 = Math.min(x0 + 1, width - 1);
                const y1 = Math.min(y0 + 1, height - 1);

                const i00 = y0 * width + x0;
                const i10 = y0 * width + x1;
                const i01 = y1 * width + x0;
                const i11 = y1 * width + x1;

                out[y * outW + x] = 0.25 * (input[i00] + input[i10] + input[i01] + input[i11]);
            }
        }

        return { data: out, width: outW, height: outH };
    }

    _buildPyramid(base, width, height) {
        const pyramid = [{ data: base, width, height }];

        while (pyramid.length < this.maxLevels) {
            const last = pyramid[pyramid.length - 1];
            if (last.width <= this.minPyramidSize || last.height <= this.minPyramidSize) {
                break;
            }
            pyramid.push(this._downsample2x(last.data, last.width, last.height));
        }

        return pyramid;
    }

    _sampleBilinear(image, width, height, x, y) {
        const cx = Math.max(0, Math.min(width - 1, x));
        const cy = Math.max(0, Math.min(height - 1, y));

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

        const v00 = image[i00];
        const v10 = image[i10];
        const v01 = image[i01];
        const v11 = image[i11];

        const vx0 = v00 + (v10 - v00) * tx;
        const vx1 = v01 + (v11 - v01) * tx;
        return vx0 + (vx1 - vx0) * ty;
    }

    _warpImage(curr, width, height, flowU, flowV) {
        const out = new Float32Array(width * height);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = y * width + x;
                const sx = x + flowU[i];
                const sy = y + flowV[i];
                out[i] = this._sampleBilinear(curr, width, height, sx, sy);
            }
        }
        return out;
    }

    _upsampleFlow(flowU, flowV, srcW, srcH, dstW, dstH) {
        const outU = new Float32Array(dstW * dstH);
        const outV = new Float32Array(dstW * dstH);

        const scaleX = dstW / srcW;
        const scaleY = dstH / srcH;

        for (let y = 0; y < dstH; y++) {
            for (let x = 0; x < dstW; x++) {
                const sx = (x + 0.5) / scaleX - 0.5;
                const sy = (y + 0.5) / scaleY - 0.5;
                const i = y * dstW + x;

                outU[i] = this._sampleBilinear(flowU, srcW, srcH, sx, sy) * scaleX;
                outV[i] = this._sampleBilinear(flowV, srcW, srcH, sx, sy) * scaleY;
            }
        }

        return { u: outU, v: outV };
    }

    _smoothFlow(flowU, flowV, width, height) {
        const outU = new Float32Array(width * height);
        const outV = new Float32Array(width * height);
        const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let su = 0;
                let sv = 0;
                let sw = 0;
                let k = 0;

                for (let dy = -1; dy <= 1; dy++) {
                    const yy = Math.max(0, Math.min(height - 1, y + dy));
                    for (let dx = -1; dx <= 1; dx++) {
                        const xx = Math.max(0, Math.min(width - 1, x + dx));
                        const w = kernel[k++];
                        const i = yy * width + xx;
                        su += flowU[i] * w;
                        sv += flowV[i] * w;
                        sw += w;
                    }
                }

                const outI = y * width + x;
                outU[outI] = su / sw;
                outV[outI] = sv / sw;
            }
        }

        return { u: outU, v: outV };
    }

    _lerp(a, b, t) {
        return a + (b - a) * t;
    }

    async _computeIncrement(prevFloat, currFloat, width, height, winSize) {
        if (!this.pipeline) {
            throw new Error('WebGPU pipeline not initialized');
        }

        this._ensureBuffers(width, height);

        const radius = Math.max(1, Math.floor(winSize / 2));

        this.device.queue.writeBuffer(this.prevBuffer, 0, prevFloat);
        this.device.queue.writeBuffer(this.currBuffer, 0, currFloat);

        const params = new Uint32Array([width, height, radius, 0]);
        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.prevBuffer } },
                { binding: 1, resource: { buffer: this.currBuffer } },
                { binding: 2, resource: { buffer: this.flowBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ]
        });

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(width / 16),
            Math.ceil(height / 16),
            1
        );
        pass.end();

        encoder.copyBufferToBuffer(
            this.flowBuffer,
            0,
            this.readBuffer,
            0,
            this.pixelCount * 2 * 4
        );

        this.device.queue.submit([encoder.finish()]);

        await this.readBuffer.mapAsync(GPUMapMode.READ);
        const mapped = this.readBuffer.getMappedRange();
        const out = new Float32Array(mapped.slice(0));
        this.readBuffer.unmap();

        return out;
    }

    async compute(prevGray, currGray, width, height, winSize) {
        // Convert input uint8 grayscale arrays to float [0, 1]
        const basePrev = new Float32Array(width * height);
        const baseCurr = new Float32Array(width * height);
        for (let i = 0; i < basePrev.length; i++) {
            basePrev[i] = prevGray[i] / 255.0;
            baseCurr[i] = currGray[i] / 255.0;
        }

        const prevPyr = this._buildPyramid(basePrev, width, height);
        const currPyr = this._buildPyramid(baseCurr, width, height);

        let flowU = null;
        let flowV = null;
        let flowW = 0;
        let flowH = 0;

        // Traverse from coarsest to finest
        const totalLevels = prevPyr.length;
        for (let level = totalLevels - 1; level >= 0; level--) {
            const p = prevPyr[level];
            const c = currPyr[level];

            if (!flowU) {
                flowU = new Float32Array(p.width * p.height);
                flowV = new Float32Array(p.width * p.height);
            } else {
                const up = this._upsampleFlow(flowU, flowV, flowW, flowH, p.width, p.height);
                flowU = up.u;
                flowV = up.v;
            }
            flowW = p.width;
            flowH = p.height;

            const levelScale = width / p.width;
            const levelWin = Math.max(3, this._toOdd(Math.round(winSize / levelScale)));
            const fineLevelIndex = (totalLevels - 1) - level;
            const levelT = totalLevels > 1 ? fineLevelIndex / (totalLevels - 1) : 1;
            const updateBlend = this._lerp(this.updateBlendCoarse, this.updateBlendFine, levelT);
            const smoothPasses = Math.max(
                0,
                Math.round(this._lerp(this.smoothPassesCoarse, this.smoothPassesFine, levelT))
            );

            for (let iter = 0; iter < this.iterationsPerLevel; iter++) {
                const warpedCurr = this._warpImage(c.data, p.width, p.height, flowU, flowV);
                const delta = await this._computeIncrement(p.data, warpedCurr, p.width, p.height, levelWin);

                for (let i = 0; i < p.width * p.height; i++) {
                    flowU[i] += delta[i * 2] * updateBlend;
                    flowV[i] += delta[i * 2 + 1] * updateBlend;
                }

                for (let s = 0; s < smoothPasses; s++) {
                    const smooth = this._smoothFlow(flowU, flowV, p.width, p.height);
                    flowU = smooth.u;
                    flowV = smooth.v;
                }
            }
        }

        for (let s = 0; s < this.finalSmoothPasses; s++) {
            const smooth = this._smoothFlow(flowU, flowV, width, height);
            flowU = smooth.u;
            flowV = smooth.v;
        }

        // Pack final dense flow for drop-in compatibility
        const packed = new Float32Array(width * height * 2);
        for (let i = 0; i < width * height; i++) {
            packed[i * 2] = flowU[i];
            packed[i * 2 + 1] = flowV[i];
        }

        return packed;
    }

    dispose() {
        if (this.prevBuffer) this.prevBuffer.destroy();
        if (this.currBuffer) this.currBuffer.destroy();
        if (this.flowBuffer) this.flowBuffer.destroy();
        if (this.readBuffer) this.readBuffer.destroy();
        if (this.paramsBuffer) this.paramsBuffer.destroy();
    }
}

window.WebGPUFarneback = WebGPUFarneback;