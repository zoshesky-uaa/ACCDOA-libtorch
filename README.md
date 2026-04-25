# ACCDOA-libtorch

A recording application that utilizes quadrophonic sound and determines object directionality via SELDnet ACCDOA model. 
Model implementation is underway, feature extraction process is developed.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zoshesky-uaa/ACCDOA-libtorch.git
   ```
2. Navigate to project directory manual or import into Visual Studio.
3. Run libtorch.ps1, libtorch.sh, or import your own pre-compiled LibTorch library to the project. I use the Nightly Debug branch of LibTorch for CUDA 13.2 which supports Visual Studio 2026.
4. Build in Visual Studio or manually with CMake:
   ```bash
	cmake --preset x64-debug
	cmake --build --preset x64-debug
   ```
5. Run the test.py script to verify functionality or utilize it as a template for your own application.