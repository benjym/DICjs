const video = document.getElementById('video');
const canvasSource = document.getElementById('canvas-source');
const ctxSource = canvasSource.getContext('2d');
const canvasOutput = document.getElementById('canvas-output');
const ctxOutput = canvasOutput.getContext('2d');
const downloadButton = document.getElementById('downloadButton');

let prevGray = null;
let streaming = false;
let width, height;
let flowData = null;
let cvLoaded = false;

let currentCameraIndex = 0;
let videoDevices = [];

// --- GUI Controls ---
const gui = new lil.GUI();
const controls = {
    flowStep: 16,
    winSize: 15,
    incremental: false,
    vectorScale: 1.0,
    resolution: '480p (640×480)',
    captureFrame: () => {
        if (streaming && cvLoaded) {
            ctxSource.drawImage(video, 0, 0, width, height);
            let imageData = ctxSource.getImageData(0, 0, width, height);

            let src = new cv.Mat(height, width, cv.CV_8UC4);
            src.data.set(imageData.data);

            let gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

            // Important: Delete the *old* prevGray before replacing it.
            if (prevGray) prevGray.delete();
            prevGray = gray;

            downloadButton.style.display = 'inline-block';
            src.delete();
        }
    }
};

// Resolution presets
const resolutionPresets = {
    'Auto (Max)': { width: 4096, height: 2160 },
    '4K (3840×2160)': { width: 3840, height: 2160 },
    '1440p (2560×1440)': { width: 2560, height: 1440 },
    '1080p (1920×1080)': { width: 1920, height: 1080 },
    '720p (1280×720)': { width: 1280, height: 720 },
    '480p (640×480)': { width: 640, height: 480 }
};

gui.add(controls, 'flowStep', 2, 64, 1).name('Flow Density');
gui.add(controls, 'winSize', 3, 256, 2).name('Window Size');
gui.add(controls, 'incremental').name('Incremental');
gui.add(controls, 'vectorScale', 0.1, 5.0, 0.1).name('Vector Scale');
// Resolution dropdown will be added after camera capabilities are known
gui.add(controls, 'captureFrame').name('Capture Frame');

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

let cameraCapabilities = null;
let initialStream = null;

function getCameras() {
    // Get selected resolution for initial request
    const selectedResolution = resolutionPresets[controls.resolution];
    let targetWidth, targetHeight;

    if (controls.resolution === 'Auto (Max)') {
        targetWidth = 4096;
        targetHeight = 2160;
    } else {
        targetWidth = selectedResolution.width;
        targetHeight = selectedResolution.height;
    }

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
            availableResolutions['Auto (Max)'] = resolutionPresets['Auto (Max)'];

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

            // if (videoDevices.length > 1) {
            //     const cameraOptions = {};
            //     videoDevices.forEach((device, index) => {
            //         cameraOptions[`Camera ${index + 1}`] = index;
            //     });

            //     gui.add({ camera: currentCameraIndex }, 'camera', cameraOptions)
            //         .name('Camera')
            //         .onChange(value => {
            //             currentCameraIndex = value;
            //             startCamera();
            //         });
            // }

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

    // Stop current processing
    if (streaming) {
        stopProcessing();
    }

    // Request new stream with exact resolution
    const constraints = {
        video: {
            deviceId: videoDevices[currentCameraIndex]?.deviceId || undefined,
            width: { ideal: targetWidth },
            height: { ideal: targetHeight },
            frameRate: { ideal: 30 }
        },
        audio: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
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
            console.error("Failed to change resolution:", err.message);
            alert("Failed to change resolution: " + err.message);
        });
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

    // Use initial stream if available, otherwise request new one (only for camera switching)
    if (initialStream && currentCameraIndex === 0) {
        console.log("Using initial high-resolution stream");
        video.srcObject = initialStream;
        video.play();
        setupVideoHandlers();
        initialStream = null; // Clear it so we don't reuse it
    } else {
        // Only request new stream when switching cameras (not for resolution changes)
        console.log("Switching to camera", currentCameraIndex);
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
            width: { ideal: targetWidth },
            height: { ideal: targetHeight },
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
            setupVideoHandlers();
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
        canvasSource.width = width;
        canvasSource.height = height;
        canvasOutput.width = width;
        canvasOutput.height = height;

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

function downloadFlowData(flow) {
    const wsU = XLSX.utils.aoa_to_sheet(flow.u);
    const wsV = XLSX.utils.aoa_to_sheet(flow.v);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, wsU, "Horizontal Displacements");
    XLSX.utils.book_append_sheet(wb, wsV, "Vertical Displacements");
    XLSX.writeFile(wb, "optical_flow_data.xlsx");
}

// --- Optical Flow Processing ---

function processVideo() {
    if (!streaming || !cvLoaded) {
        setTimeout(processVideo, 0);
        return;
    }

    let begin = Date.now();

    try {
        // console.log("cap state:", cap);
        // console.log("frame before read: size=", frame.size());

        // Attempt to read the frame
        cap.read(frame);

        // console.log("frame after read: size=", frame.size());

        // Convert the current frame to grayscale.
        cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

        // If prevGray is not null, calculate optical flow.
        if (prevGray) {
            let winSize = controls.winSize % 2 === 0 ? controls.winSize + 1 : controls.winSize;
            cv.calcOpticalFlowFarneback(prevGray, gray, flow, 0.5, 3, winSize, 3, 5, 1.2, 0);

            // --- Store Flow Data ---
            let uData = [];
            let vData = [];
            for (let y = 0; y < height; y++) {
                let uRow = [];
                let vRow = [];
                for (let x = 0; x < width; x++) {
                    let fx = flow.data32F[y * flow.cols * 2 + x * 2];
                    let fy = flow.data32F[y * flow.cols * 2 + x * 2 + 1];
                    uRow.push(fx);
                    vRow.push(fy);
                }
                uData.push(uRow);
                vData.push(vRow);
            }
            flowData = { u: uData, v: vData };

            // --- Visualization ---
            // Draw the *current* video frame onto the output canvas.
            ctxOutput.drawImage(video, 0, 0, width, height);
            // ctxOutput.strokeStyle = 'white';
            ctxOutput.strokeStyle = 'black';
            ctxOutput.lineWidth = 2;

            // Draw the flow vectors.
            for (let y = 0; y < height; y += controls.flowStep) {
                for (let x = 0; x < width; x += controls.flowStep) {
                    let fx = flow.data32F[y * flow.cols * 2 + x * 2];
                    let fy = flow.data32F[y * flow.cols * 2 + x * 2 + 1];

                    // Apply vector scaling
                    let scaledFx = fx * controls.vectorScale;
                    let scaledFy = fy * controls.vectorScale;

                    ctxOutput.beginPath();
                    ctxOutput.moveTo(x, y);
                    ctxOutput.lineTo(x + scaledFx, y + scaledFy);
                    ctxOutput.stroke();
                }
            }

        } else {
            // If no prevGray, still draw the video
            ctxOutput.drawImage(video, 0, 0, width, height);
        }

        if (controls.incremental) {
            prevGray = gray.clone(); // Clone current gray to prevGray
        }

    } catch (err) {
        console.error("Error in processVideo:", err);

        // JUST FOR DEBUGGING
        stopProcessing(); // Stop the loop on error
        return; // Exit the function to prevent further execution
    }

    let delay = Math.max(0, 1000 / 30 - (Date.now() - begin));
    setTimeout(processVideo, delay);
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

// Call getCameras on page load to populate the dropdown
getCameras();