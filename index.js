const video = document.getElementById('video');
const canvasOutput = document.getElementById('canvas-output');
const ctxOutput = canvasOutput.getContext('2d');
const downloadButton = document.getElementById('downloadButton');
const screenshotButton = document.getElementById('screenshotButton');

let prevGray = null;
let streaming = false;
let width, height;
let flowData = null;
let cvLoaded = false;
let webGpuFarneback = null;
let webGpuFarnebackReady = false;

let showWindowSizePreview = false;
let windowSizePreviewTimeout = null;

let currentCameraIndex = 0;
let videoDevices = [];

// --- GUI Controls ---
const gui = new lil.GUI();
const controls = {
    // method: 'Lucas-Kanade',
    method: 'Farneback (WebGPU)',
    flowStep: 16,
    winSize: 15,
    incremental: false,
    vectorScale: 1.0,
    resolution: '480p (640×480)',
    arrowColour: '##000000',
    captureFrame: () => {
        if (streaming && cvLoaded) {
            // Create a temporary canvas to capture the current frame without displaying it
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const tempCtx = tempCanvas.getContext('2d');

            tempCtx.drawImage(video, 0, 0, width, height);
            let imageData = tempCtx.getImageData(0, 0, width, height);

            let src = new cv.Mat(height, width, cv.CV_8UC4);
            src.data.set(imageData.data);

            let gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

            // Important: Delete the *old* prevGray before replacing it.
            if (prevGray) prevGray.delete();
            prevGray = gray;

            downloadButton.style.display = 'inline-block';
            screenshotButton.style.display = 'inline-block';
            src.delete();
        }
    }
};

// Resolution presets
const resolutionPresets = {
    // 'Auto (Max)': { width: 4096, height: 2160 },
    '4K (3840×2160)': { width: 3840, height: 2160 },
    '1440p (2560×1440)': { width: 2560, height: 1440 },
    '1080p (1920×1080)': { width: 1920, height: 1080 },
    '720p (1280×720)': { width: 1280, height: 720 },
    '480p (640×480)': { width: 640, height: 480 }
};

// gui.add(controls, 'method', ['Lucas-Kanade', 'Farneback', 'Farneback (WebGPU)']).name('Method');
gui.add(controls, 'flowStep', 2, 64, 1).name('Flow Density');
gui.add(controls, 'winSize', 3, 256, 2).name('Window Size').onChange(() => {
    showWindowSizePreview = true;
    clearTimeout(windowSizePreviewTimeout);
    windowSizePreviewTimeout = setTimeout(() => { showWindowSizePreview = false; }, 1500);
});
gui.add(controls, 'vectorScale', 0.1, 5.0, 0.1).name('Vector Scale');
gui.addColor(controls, 'arrowColour').name('Vector Colour');
gui.add(controls, 'incremental').name('Incremental');
gui.add(controls, 'captureFrame').name('Capture Reference Frame');
// Resolution dropdown will be added after camera capabilities are known

// --- OpenCV and Video Setup ---
let cap;
let frame;      // Current frame Mat
let gray;       // Current frame grayscale Mat
let flow;       // Optical flow Mat

function onOpenCvReady() {
    cv['onRuntimeInitialized'] = () => {
        cvLoaded = true;
        console.log("OpenCV.js is ready.");
        // Only start camera if devices are already enumerated
        if (videoDevices.length > 0) {
            startCamera();
        }
    };
}

async function initWebGpuFarneback() {
    if (typeof WebGPUFarneback === 'undefined') return;
    try {
        webGpuFarneback = await WebGPUFarneback.create();
        webGpuFarnebackReady = true;
        console.log('WebGPU Farneback is ready.');
    } catch (err) {
        webGpuFarnebackReady = false;
        console.warn('WebGPU Farneback unavailable:', err.message);
    }
}

let cameraCapabilities = null;
let initialStream = null;
let cameraOptions = {};
let orientationAngle = 0;
let cameraFacingMode = 'environment'; // 'user' for front-facing, 'environment' for back-facing

function getDeviceOrientationAngle() {
    let angle = 0;
    if (screen.orientation && typeof screen.orientation.angle === 'number') {
        angle = screen.orientation.angle;
    } else if (typeof window.orientation === 'number') {
        angle = window.orientation;
    }
    angle = ((angle % 360) + 360) % 360;
    return [0, 90, 180, 270].includes(angle) ? angle : 0;
}

function getRotationAngle(baseAngle) {
    // Adjust rotation angle based on camera facing
    if (cameraFacingMode === 'user') {
        // Front-facing camera: adjust rotation for mirror orientation
        return (360 - baseAngle) % 360;
    }
    return baseAngle;
}

function applyOrientationRotation() {
    orientationAngle = getDeviceOrientationAngle();
    if (!width || !height) return;

    if (orientationAngle === 90 || orientationAngle === 270) {
        canvasOutput.width = height;
        canvasOutput.height = width;
    } else {
        canvasOutput.width = width;
        canvasOutput.height = height;
    }

    canvasOutput.style.width = '100%';
    canvasOutput.style.height = 'auto';
    canvasOutput.style.aspectRatio = 'auto';
}

function handleOrientationChange() {
    const newAngle = getDeviceOrientationAngle();
    if (newAngle === orientationAngle) return;
    console.log('Device orientation changed to', newAngle);
    applyOrientationRotation();
}

function applyCanvasRotation(ctx, angle) {
    // Apply rotation with translation based on the final display angle
    // Account for camera facing mode when determining the rotation
    const adjustedAngle = getRotationAngle(angle);

    if (adjustedAngle === 270) {
        ctx.translate(canvasOutput.width, 0);
        ctx.rotate(Math.PI / 2);
    } else if (adjustedAngle === 180) {
        ctx.translate(canvasOutput.width, canvasOutput.height);
        ctx.rotate(Math.PI);
    } else if (adjustedAngle === 90) {
        ctx.translate(0, canvasOutput.height);
        ctx.rotate(-Math.PI / 2);
    }
}

function getCameras() {
    // Get selected resolution for initial request
    const selectedResolution = resolutionPresets[controls.resolution];
    let targetWidth, targetHeight;

    // if (controls.resolution === 'Auto (Max)') {
    //     targetWidth = 4096;
    //     targetHeight = 2160;
    // } else {
    targetWidth = selectedResolution.width;
    targetHeight = selectedResolution.height;
    // }

    // Request camera access with selected resolution
    const constraints = {
        video: {
            width: { ideal: targetWidth },
            height: { ideal: targetHeight },
            frameRate: { ideal: 30 }
        },
        audio: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            initialStream = stream;
            // Get the video track to check capabilities and enumerate devices
            const videoTrack = stream.getVideoTracks()[0];
            cameraCapabilities = videoTrack.getCapabilities();
            cameraFacingMode = videoTrack.getSettings().facingMode || 'environment';
            console.log('Camera facing mode:', cameraFacingMode);

            console.log("Camera capabilities:", cameraCapabilities);
            console.log("Initial stream resolution:", videoTrack.getSettings());

            // Now enumerate devices
            return navigator.mediaDevices.enumerateDevices();
        })
        .then(devices => {
            videoDevices = devices.filter(device => device.kind === 'videoinput');
            console.log("Video devices:", videoDevices);

            // Filter resolution presets based on camera capabilities
            const availableResolutions = {};
            const maxWidth = cameraCapabilities?.width?.max || 4096;
            const maxHeight = cameraCapabilities?.height?.max || 2160;

            console.log("Camera max resolution:", maxWidth, "x", maxHeight);

            // Always include Auto (Max)
            // availableResolutions['Auto (Max)'] = resolutionPresets['Auto (Max)'];

            // Filter other resolutions that are within camera capabilities
            Object.entries(resolutionPresets).forEach(([name, res]) => {
                if (name !== 'Auto (Max)' && res.width <= maxWidth && res.height <= maxHeight) {
                    availableResolutions[name] = res;
                }
            });

            console.log("Available resolutions:", Object.keys(availableResolutions));

            // Add resolution dropdown with filtered options
            gui.add(controls, 'resolution', Object.keys(availableResolutions))
                .name('Resolution')
                .onChange(value => {
                    console.log('Resolution changed to:', value);
                    changeResolution(value);
                });


            //console.log(videoDevices) ;
            {
                videoDevices.forEach((device, index) => {
                    cameraOptions[`Camera ${index + 1}`] = index;
                });

                gui.add({ camera: currentCameraIndex }, 'camera', cameraOptions)
                    .name('Camera')
                    .onChange(value => {
                        currentCameraIndex = value;
                        startCamera();
                    });
            }

            // Only start camera after OpenCV is ready AND devices are enumerated
            if (cvLoaded) {
                startCamera();
            }
        })
        .catch(err => {
            console.error('Error accessing camera:', err);
            alert('Camera permissions are required to proceed.');
        });
}

function changeResolution(resolutionName) {
    if (!video.srcObject) {
        console.log("No video stream to change resolution on");
        return;
    }

    // First try to change constraints on existing track
    const videoTrack = video.srcObject.getVideoTracks()[0];
    if (videoTrack) {
        console.log("Current track settings:", videoTrack.getSettings());
        console.log("Current track constraints:", videoTrack.getConstraints());
    }

    const selectedResolution = resolutionPresets[resolutionName];
    let targetWidth, targetHeight;

    if (resolutionName === 'Auto (Max)') {
        targetWidth = cameraCapabilities?.width?.max || 4096;
        targetHeight = cameraCapabilities?.height?.max || 2160;
    } else {
        targetWidth = selectedResolution.width;
        targetHeight = selectedResolution.height;
    }

    console.log("Changing resolution to:", targetWidth, "x", targetHeight);
    console.log("Current video resolution:", video.videoWidth, "x", video.videoHeight);

    // Stop current processing
    if (streaming) {
        stopProcessing();
    }

    // Request new stream with exact resolution constraints
    const constraints = {
        video: {
            deviceId: videoDevices[currentCameraIndex]?.deviceId || undefined,
            width: { exact: targetWidth },
            height: { exact: targetHeight },
            frameRate: { ideal: 30 }
        },
        audio: false
    };

    // First try applying constraints to existing track
    if (videoTrack) {
        console.log("Trying to apply constraints to existing track...");
        videoTrack.applyConstraints({
            width: { exact: targetWidth },
            height: { exact: targetHeight }
        }).then(() => {
            console.log("Successfully applied constraints to existing track");
            console.log("Updated track settings:", videoTrack.getSettings());

            // Force video element to update its dimensions
            setTimeout(() => {
                updateVideoDisplay();
            }, 50);
        }).catch(err => {
            console.log("Failed to apply constraints to existing track:", err.message);
            console.log("Requesting new stream instead...");
            requestNewVideoStream();
        });
    } else {
        requestNewVideoStream();
    }

    function requestNewVideoStream() {
        console.log("Requesting stream with constraints:", constraints);

        navigator.mediaDevices.getUserMedia(constraints)
            .then(stream => {
                console.log("Successfully got new stream with exact constraints");
                const videoTrack = stream.getVideoTracks()[0];
                cameraFacingMode = videoTrack.getSettings().facingMode || 'environment';
                console.log('Camera facing mode:', cameraFacingMode);
                console.log("New stream settings:", videoTrack.getSettings());

                // Stop the old stream
                if (video.srcObject) {
                    video.srcObject.getTracks().forEach(track => track.stop());
                }

                // Set new stream
                video.srcObject = stream;
                video.play();

                // Set up handlers for new stream
                setupVideoHandlers();
            })
            .catch(err => {
                console.error("Failed with exact constraints:", err.message);
                console.log("Trying with ideal constraints as fallback");

                // Fallback to ideal constraints
                const fallbackConstraints = {
                    video: {
                        deviceId: videoDevices[currentCameraIndex]?.deviceId || undefined,
                        width: { ideal: targetWidth },
                        height: { ideal: targetHeight },
                        frameRate: { ideal: 30 }
                    },
                    audio: false
                };

                return navigator.mediaDevices.getUserMedia(fallbackConstraints);
            })
            .then(stream => {
                if (stream) {
                    console.log("Successfully got fallback stream with ideal constraints");
                    const videoTrack = stream.getVideoTracks()[0];
                    cameraFacingMode = videoTrack.getSettings().facingMode || 'environment';
                    console.log('Camera facing mode:', cameraFacingMode);
                    console.log("Fallback stream settings:", videoTrack.getSettings());

                    // Stop the old stream
                    if (video.srcObject) {
                        video.srcObject.getTracks().forEach(track => track.stop());
                    }

                    // Set new stream
                    video.srcObject = stream;
                    video.play();

                    // Set up handlers for new stream
                    setupVideoHandlers();
                }
            })
            .catch(err => {
                console.error("Failed to change resolution completely:", err.message);
                alert("Failed to change resolution: " + err.message);
            });
    }
}

function updateVideoDisplay() {
    // Force the video element to recognize the new dimensions
    const videoTrack = video.srcObject.getVideoTracks()[0];
    const settings = videoTrack.getSettings();

    console.log("Updating video display for resolution:", settings.width, "x", settings.height);

    // Stop current processing first
    if (streaming) {
        stopProcessing();
    }

    // Update global width/height variables
    width = settings.width;
    height = settings.height;

    // Update video element dimensions
    video.width = width;
    video.height = height;
    video.style.aspectRatio = `${width} / ${height}`;

    // Update canvas dimensions and apply orientation
    canvasOutput.width = width;
    canvasOutput.height = height;
    applyOrientationRotation();

    // Reinitialize OpenCV Mats with new dimensions
    if (frame) frame.delete();
    if (gray) gray.delete();
    if (flow) flow.delete();

    // Clear the reference frame since it has old dimensions
    if (prevGray) {
        console.log("Clearing reference frame due to resolution change");
        prevGray.delete();
        prevGray = null;
        // Hide download button since reference frame is cleared
        downloadButton.style.display = 'none';
        screenshotButton.style.display = 'none';
    }

    cap = new cv.VideoCapture(video);
    frame = new cv.Mat(height, width, cv.CV_8UC4);
    gray = new cv.Mat(height, width, cv.CV_8UC1);
    flow = new cv.Mat();

    console.log("Updated OpenCV Mats for dimensions:", width, "x", height);
    console.log("Frame Mat size:", frame.cols, "x", frame.rows);

    // Restart video processing
    streaming = true;
    setTimeout(processVideo, 0);
}

function startCamera() {
    if (!cvLoaded) return;

    console.log("Starting camera, current streaming state:", streaming);

    // Always stop processing first when switching cameras
    if (streaming) {
        console.log("Stopping current processing before camera switch");
        stopProcessing();
    }

    // Clean up any existing video stream
    if (video.srcObject && !initialStream) {
        console.log("Cleaning up existing video stream");
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    // For the first time setup only, use initial stream if available
    if (!video.srcObject && initialStream && currentCameraIndex === 0) {
        console.log("Using initial high-resolution stream for first setup");
        video.srcObject = initialStream;
        video.play();
        setupVideoHandlers();
        initialStream = null; // Clear it so we don't reuse it
    } else {
        // Request new stream for camera switching or when video is already running
        console.log("Requesting new stream for camera", currentCameraIndex);
        // Add a small delay to ensure processing has fully stopped
        setTimeout(() => {
            requestNewStreamForCamera();
        }, 100);
    }
}

function requestNewStreamForCamera() {
    // Get current resolution setting
    const selectedResolution = resolutionPresets[controls.resolution];
    let targetWidth, targetHeight;

    if (controls.resolution === 'Auto (Max)') {
        targetWidth = cameraCapabilities?.width?.max || 4096;
        targetHeight = cameraCapabilities?.height?.max || 2160;
    } else {
        targetWidth = selectedResolution.width;
        targetHeight = selectedResolution.height;
    }

    const constraints = {
        video: {
            deviceId: videoDevices[currentCameraIndex]?.deviceId || undefined,
            width: { exact: targetWidth },
            height: { exact: targetHeight },
            frameRate: { ideal: 30 }
        },
        audio: false
    };

    console.log("Requesting new stream for camera with resolution:", targetWidth, "x", targetHeight);

    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            // Stop the old stream
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }
            video.srcObject = stream;
            video.play();
            const videoTrack = stream.getVideoTracks()[0];
            cameraFacingMode = videoTrack.getSettings().facingMode || 'environment';
            console.log('Camera facing mode:', cameraFacingMode);
            setupVideoHandlers();
        })
        .catch(err => {
            console.error("Error with exact constraints, trying ideal:", err.message);
            // Fallback to ideal constraints if exact fails
            const fallbackConstraints = {
                video: {
                    deviceId: videoDevices[currentCameraIndex]?.deviceId || undefined,
                    width: { ideal: targetWidth },
                    height: { ideal: targetHeight },
                    frameRate: { ideal: 30 }
                },
                audio: false
            };

            return navigator.mediaDevices.getUserMedia(fallbackConstraints);
        })
        .then(stream => {
            if (stream) {
                // Stop the old stream
                if (video.srcObject) {
                    video.srcObject.getTracks().forEach(track => track.stop());
                }
                video.srcObject = stream;
                video.play();
                const videoTrack = stream.getVideoTracks()[0];
                cameraFacingMode = videoTrack.getSettings().facingMode || 'environment';
                console.log('Camera facing mode:', cameraFacingMode);
                setupVideoHandlers();
            }
        })
        .catch(err => {
            console.error("Error accessing the camera:", err);
            alert("Error accessing camera: " + err.message);
        });
}

function setupVideoHandlers() {
    video.onloadedmetadata = () => {
        // Dynamically set the initial size of the video element
        video.width = video.videoWidth;
        video.height = video.videoHeight;

        console.log(video.width, video.height)

        // Use videoWidth and videoHeight for consistent dimensions
        width = video.videoWidth;
        height = video.videoHeight;

        console.log("Camera resolution:", width, "x", height);

        // Check if width < height and flip them internally
        // if (width < height) {
        //     [width, height] = [height, width];
        // }

        // Dynamically set the aspect ratio of the video element
        video.style.aspectRatio = `${width} / ${height}`;



        // console.log("video width: ", width, "video height: ", height);

        // Reinitialize canvas dimensions
        // canvasSource.width = width;
        // canvasSource.height = height;
        canvasOutput.width = width;
        canvasOutput.height = height;
        applyOrientationRotation();

        // Reinitialize Mats with correct dimensions
        if (frame) frame.delete();
        if (gray) gray.delete();
        if (flow) flow.delete();

        cap = new cv.VideoCapture(video);
        frame = new cv.Mat(height, width, cv.CV_8UC4);
        gray = new cv.Mat(height, width, cv.CV_8UC1);
        flow = new cv.Mat();

        console.log("OpenCV Mats initialized with dimensions:", width, "x", height);
        console.log("Frame Mat size:", frame.cols, "x", frame.rows);

        streaming = true;
        setTimeout(processVideo, 0); // Start processing
    };
}

downloadButton.addEventListener('click', () => {
    if (flowData) {
        downloadFlowData(flowData);
    }
});

screenshotButton.addEventListener('click', () => {
    // Create a temporary canvas to capture the current frame
    const screenshotCanvas = document.createElement('canvas');
    screenshotCanvas.width = canvasOutput.width;
    screenshotCanvas.height = canvasOutput.height;
    const screenshotCtx = screenshotCanvas.getContext('2d');
    screenshotCtx.drawImage(canvasOutput, 0, 0);

    // Convert to blob and download
    screenshotCanvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `DIC_screenshot_${width}x${height}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 'image/png');
});

function downloadFlowData(flow) {
    const wsU = XLSX.utils.aoa_to_sheet(flow.u);
    const wsV = XLSX.utils.aoa_to_sheet(flow.v);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, wsU, "Horizontal Displacements");
    XLSX.utils.book_append_sheet(wb, wsV, "Vertical Displacements");
    XLSX.writeFile(wb, "optical_flow_data.xlsx");
}

// --- Optical Flow Processing ---

async function processVideo() {
    if (!streaming || !cvLoaded) {
        setTimeout(processVideo, 0);
        return;
    }

    let begin = Date.now();

    try {
        // Attempt to read the frame
        cap.read(frame);

        // Convert the current frame to grayscale.
        cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

        // If prevGray is not null, calculate optical flow.
        if (prevGray) {
            // Safety check: ensure prevGray and gray have the same dimensions
            if (prevGray.rows !== gray.rows || prevGray.cols !== gray.cols) {
                console.log("Dimension mismatch detected, clearing reference frame");
                prevGray.delete();
                prevGray = null;
                downloadButton.style.display = 'none';
                screenshotButton.style.display = 'none';
                // Just draw the video without flow calculation
                ctxOutput.setTransform(1, 0, 0, 1, 0, 0);
                ctxOutput.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
                ctxOutput.save();
                applyCanvasRotation(ctxOutput, orientationAngle);
                ctxOutput.drawImage(video, 0, 0, width, height);
                ctxOutput.restore();
            } else {
                const step = controls.flowStep;
                let winSize = controls.winSize % 2 === 0 ? controls.winSize + 1 : controls.winSize;

                if (controls.method === 'Lucas-Kanade') {
                    // --- Lucas-Kanade sparse optical flow ---
                    const cols = Math.floor(width / step);
                    const rows = Math.floor(height / step);
                    const nPts = cols * rows;

                    let prevPts = new cv.Mat(nPts, 1, cv.CV_32FC2);
                    let i = 0;
                    for (let by = 0; by < rows; by++) {
                        for (let bx = 0; bx < cols; bx++) {
                            prevPts.data32F[i * 2] = bx * step + step / 2;
                            prevPts.data32F[i * 2 + 1] = by * step + step / 2;
                            i++;
                        }
                    }

                    let nextPts = new cv.Mat();
                    let status = new cv.Mat();
                    let err = new cv.Mat();
                    const criteria = new cv.TermCriteria(
                        cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01
                    );
                    cv.calcOpticalFlowPyrLK(
                        prevGray, gray, prevPts, nextPts, status, err,
                        new cv.Size(winSize, winSize), 3, criteria
                    );

                    // --- Store Flow Data ---
                    let uData = [];
                    let vData = [];
                    for (let by = 0; by < rows; by++) {
                        let uRow = [];
                        let vRow = [];
                        for (let bx = 0; bx < cols; bx++) {
                            const idx = by * cols + bx;
                            if (status.data[idx]) {
                                uRow.push(nextPts.data32F[idx * 2] - prevPts.data32F[idx * 2]);
                                vRow.push(nextPts.data32F[idx * 2 + 1] - prevPts.data32F[idx * 2 + 1]);
                            } else {
                                uRow.push(0);
                                vRow.push(0);
                            }
                        }
                        uData.push(uRow);
                        vData.push(vRow);
                    }
                    flowData = { u: uData, v: vData };

                    // --- Visualization ---
                    ctxOutput.setTransform(1, 0, 0, 1, 0, 0);
                    ctxOutput.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
                    ctxOutput.save();
                    applyCanvasRotation(ctxOutput, orientationAngle);
                    ctxOutput.drawImage(video, 0, 0, width, height);
                    ctxOutput.strokeStyle = controls.arrowColour;
                    ctxOutput.lineWidth = 2;

                    for (let idx = 0; idx < nPts; idx++) {
                        if (!status.data[idx]) continue;
                        const x0 = prevPts.data32F[idx * 2];
                        const y0 = prevPts.data32F[idx * 2 + 1];
                        const fx = (nextPts.data32F[idx * 2] - x0) * controls.vectorScale;
                        const fy = (nextPts.data32F[idx * 2 + 1] - y0) * controls.vectorScale;
                        ctxOutput.beginPath();
                        ctxOutput.moveTo(x0, y0);
                        ctxOutput.lineTo(x0 + fx, y0 + fy);
                        ctxOutput.stroke();
                    }
                    ctxOutput.restore();

                    prevPts.delete(); nextPts.delete(); status.delete(); err.delete();

                } else if (controls.method === 'Farneback (WebGPU)') {
                    if (!webGpuFarnebackReady || !webGpuFarneback) {
                        console.warn('WebGPU Farneback not ready, falling back to OpenCV Farneback');
                        controls.method = 'Farneback';
                    } else {
                        const flowArray = await webGpuFarneback.compute(
                            prevGray.data,
                            gray.data,
                            width,
                            height,
                            winSize
                        );

                        // --- Store Flow Data ---
                        let uData = [];
                        let vData = [];
                        for (let y = 0; y < height; y++) {
                            let uRow = [];
                            let vRow = [];
                            const rowStart = y * width * 2;
                            for (let x = 0; x < width; x++) {
                                uRow.push(flowArray[rowStart + x * 2]);
                                vRow.push(flowArray[rowStart + x * 2 + 1]);
                            }
                            uData.push(uRow);
                            vData.push(vRow);
                        }
                        flowData = { u: uData, v: vData };

                        // --- Visualization ---
                        ctxOutput.setTransform(1, 0, 0, 1, 0, 0);
                        ctxOutput.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
                        ctxOutput.save();
                        applyCanvasRotation(ctxOutput, orientationAngle);
                        ctxOutput.drawImage(video, 0, 0, width, height);
                        ctxOutput.strokeStyle = controls.arrowColour;
                        ctxOutput.lineWidth = 2;

                        for (let y = 0; y < height; y += step) {
                            for (let x = 0; x < width; x += step) {
                                const idx = (y * width + x) * 2;
                                const fx = flowArray[idx] * controls.vectorScale;
                                const fy = flowArray[idx + 1] * controls.vectorScale;
                                ctxOutput.beginPath();
                                ctxOutput.moveTo(x, y);
                                ctxOutput.lineTo(x + fx, y + fy);
                                ctxOutput.stroke();
                            }
                        }
                        ctxOutput.restore();
                    }
                }

                if (controls.method === 'Farneback') {
                    // --- Farneback dense optical flow ---
                    cv.calcOpticalFlowFarneback(prevGray, gray, flow, 0.5, 3, winSize, 3, 5, 1.2, 0);

                    // --- Store Flow Data ---
                    let uData = [];
                    let vData = [];
                    for (let y = 0; y < height; y++) {
                        let uRow = [];
                        let vRow = [];
                        for (let x = 0; x < width; x++) {
                            uRow.push(flow.data32F[y * flow.cols * 2 + x * 2]);
                            vRow.push(flow.data32F[y * flow.cols * 2 + x * 2 + 1]);
                        }
                        uData.push(uRow);
                        vData.push(vRow);
                    }
                    flowData = { u: uData, v: vData };

                    // --- Visualization ---
                    ctxOutput.setTransform(1, 0, 0, 1, 0, 0);
                    ctxOutput.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
                    ctxOutput.save();
                    applyCanvasRotation(ctxOutput, orientationAngle);
                    ctxOutput.drawImage(video, 0, 0, width, height);
                    ctxOutput.strokeStyle = controls.arrowColour;
                    ctxOutput.lineWidth = 2;

                    for (let y = 0; y < height; y += step) {
                        for (let x = 0; x < width; x += step) {
                            const fx = flow.data32F[y * flow.cols * 2 + x * 2] * controls.vectorScale;
                            const fy = flow.data32F[y * flow.cols * 2 + x * 2 + 1] * controls.vectorScale;
                            ctxOutput.beginPath();
                            ctxOutput.moveTo(x, y);
                            ctxOutput.lineTo(x + fx, y + fy);
                            ctxOutput.stroke();
                        }
                    }
                    ctxOutput.restore();
                }
            }
        } else {
            // If no prevGray, still draw the video
            ctxOutput.setTransform(1, 0, 0, 1, 0, 0);
            ctxOutput.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
            ctxOutput.save();
            applyCanvasRotation(ctxOutput, orientationAngle);
            ctxOutput.drawImage(video, 0, 0, width, height);
            ctxOutput.restore();
        }

        if (controls.incremental) {
            prevGray = gray.clone(); // Clone current gray to prevGray
        }

        drawWindowSizeOverlay();

    } catch (err) {
        console.error("Error in processVideo:", err);

        // JUST FOR DEBUGGING
        stopProcessing(); // Stop the loop on error
        return; // Exit the function to prevent further execution
    }

    let delay = Math.max(0, 1000 / 30 - (Date.now() - begin));
    setTimeout(processVideo, delay);
}
function drawWindowSizeOverlay() {
    if (!showWindowSizePreview || !width || !height) return;
    const size = controls.winSize % 2 === 0 ? controls.winSize + 1 : controls.winSize;
    const cx = Math.round(canvasOutput.width / 2);
    const cy = Math.round(canvasOutput.height / 2);
    ctxOutput.save();
    ctxOutput.setTransform(1, 0, 0, 1, 0, 0);
    ctxOutput.strokeStyle = 'rgba(255, 220, 0, 0.95)';
    ctxOutput.lineWidth = 2;
    ctxOutput.strokeRect(cx - size / 2, cy - size / 2, size, size);
    ctxOutput.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctxOutput.fillRect(cx - size / 2, cy - size / 2 - 22, size < 80 ? 80 : size, 20);
    ctxOutput.fillStyle = 'rgba(255, 220, 0, 0.95)';
    ctxOutput.font = 'bold 13px monospace';
    ctxOutput.fillText(`Window: ${size}×${size}px`, cx - size / 2 + 4, cy - size / 2 - 7);
    ctxOutput.restore();
}

function stopProcessing() {
    console.log("Stopping processing, current streaming state:", streaming);
    if (streaming) {
        streaming = false;
        console.log("Processing stopped");
    }

    // Also clean up any pending timeouts by giving them a moment to check the streaming flag
    setTimeout(() => {
        console.log("Processing cleanup complete");
    }, 50);
}

window.addEventListener('orientationchange', handleOrientationChange);
if (screen.orientation && typeof screen.orientation.addEventListener === 'function') {
    screen.orientation.addEventListener('change', handleOrientationChange);
}
window.addEventListener('resize', handleOrientationChange);

// Call getCameras on page load to populate the dropdown
initWebGpuFarneback();
getCameras();
