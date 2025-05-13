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
gui.add(controls, 'flowStep', 2, 64, 1).name('Flow Density');
gui.add(controls, 'winSize', 3, 31, 2).name('Window Size');
gui.add(controls, 'incremental').name('Incremental');
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
        startCamera();
    };
}

function getCameras() {
    // Request camera access to ensure permissions are granted
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(() => {
            navigator.mediaDevices.enumerateDevices()
                .then(devices => {
                    videoDevices = devices.filter(device => device.kind === 'videoinput');

                    if (videoDevices.length > 1) {
                        const cameraOptions = {};
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

                    // Start the camera after devices are enumerated
                    startCamera();
                })
                .catch(err => {
                    console.error('Error enumerating devices:', err);
                });
        })
        .catch(err => {
            console.error('Error accessing camera:', err);
            alert('Camera permissions are required to proceed.');
        });
}

function startCamera() {
    if (!cvLoaded) return;

    if (streaming) {
        stopProcessing();
    }

    const constraints = {
        video: {
            deviceId: videoDevices[currentCameraIndex]?.deviceId || undefined
        },
        audio: false
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            video.srcObject = stream;
            video.play();
            video.onloadedmetadata = () => {
                // Use videoWidth and videoHeight for consistent dimensions
                width = video.videoWidth;
                height = video.videoHeight;
                console.log("video width: ", width, "video height: ", height);

                // Reinitialize canvas dimensions
                canvasSource.width = width;
                canvasSource.height = height;
                canvasOutput.width = width;
                canvasOutput.height = height;

                // Reinitialize Mats with correct dimensions
                if (cap) cap.delete();
                if (frame) frame.delete();
                if (gray) gray.delete();
                if (flow) flow.delete();

                cap = new cv.VideoCapture(video);
                frame = new cv.Mat(height, width, cv.CV_8UC4);
                gray = new cv.Mat(height, width, cv.CV_8UC1);
                flow = new cv.Mat();

                streaming = true;
                setTimeout(processVideo, 0); // Start processing
            };
        })
        .catch(err => {
            console.error("Error accessing the camera:", err);
            alert("Error accessing camera: " + err.message);
        });
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
        console.log(cap)
        console.log("Frame size: ", frame.size());
        cap.read(frame); // Read the frame  

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
            ctxOutput.strokeStyle = 'white';
            ctxOutput.lineWidth = 2;

            // Draw the flow vectors.
            for (let y = 0; y < height; y += controls.flowStep) {
                for (let x = 0; x < width; x += controls.flowStep) {
                    let fx = flow.data32F[y * flow.cols * 2 + x * 2];
                    let fy = flow.data32F[y * flow.cols * 2 + x * 2 + 1];

                    ctxOutput.beginPath();
                    ctxOutput.moveTo(x, y);
                    ctxOutput.lineTo(x + fx, y + fy);
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
    if (streaming) {
        streaming = false;
    }
}

// Call getCameras on page load to populate the dropdown
getCameras();