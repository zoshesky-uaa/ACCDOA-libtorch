# ACCDOA-libtorch

A recording application that utilizes quadrophonic sound and determines object directionality via SELDnet ACCDOA model. 
Model implementation is underway, feature extraction process is developed.

## Building
__Setup__: Adjust the CMake settings to your system capabilities, including CUDA Toolkit and cuDNN.
1. Clone the repository:
   ```bash
   git clone https://github.com/zoshesky-uaa/ACCDOA-libtorch.git
   ```
2. Navigate to project directory.
3. Run libtorch.ps1, libtorch.sh, or import your own pre-compiled LibTorch library to the project. I use the Nightly Debug branch of LibTorch for CUDA 13.
4. Adjust the ``CMakePresets.json`` to your configuration, you will need the follow softwares configured:
   * Clang or Visual Studio with clangd installed.
   * vcpkg or Visual Studio with vcpkg installed.
   * NVIDIA Cuda Toolkit (12.8+ or better, whatever works with your Visual Studio if you want intergration)
   * NVIDIA cuDNN
4. Build and install the project with CMake:
   ```bash
	cmake --preset x64-debug
	cmake --build --preset x64-debug --target install
   ```
   Note that adjustments might need to be made to ``CMakeLists.txt`` regarding the CPU architecture if it doesn't support (x86) AVX2 (-mavx2) or prehaps support AVX512. Please consult KFR's CMake compiling settings regarding tuning to so it best supports your system architecture.

6. Run the ``test_generate.py`` script to verify functionality or utilize it as a template for your own application.

The final compiled packages should be present in ``/out/install