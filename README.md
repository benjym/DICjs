# DICjs

This project demonstrates the Dense Optical Flow algorithm (the Lucas Kanade implementation of Digital Image Correlation) from OpenCV.js in a web browser, with interactive controls via lil-gui and the ability to download displacement data as an XLSX file.

## Cross-validation script

To compare OpenCV Farneback and the custom WebGPU Farneback against known synthetic displacements:

1. Open validate-farneback.html in a browser that supports WebGPU.
2. Click Run Validation.
3. Inspect the reported metrics:
	- OpenCV MAE/RMSE against ground truth
	- WebGPU MAE/RMSE against ground truth
	- OpenCV vs WebGPU mean/RMS difference

The script uses generated translation fields (dx, dy) so expected flow is known analytically.

## Usage

To use this project, follow these steps:

1. [Open the HTML file in a web browser](https://benjym.github.io/DICjs/).
2. Interact with the controls provided by lil-gui to adjust parameters.
3. Click the download button to save the displacement data as an XLSX file.