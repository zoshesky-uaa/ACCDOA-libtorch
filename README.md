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

The final compiled packages should be present in ``/out/install``.

## Dependencies

The build pulls libraries from three places. If a configure step fails, check which bucket the missing library belongs to before changing CMake.

**Resolved by vcpkg (manifest mode, triplet `x64-windows`, see `vcpkg.json`):**
* `blosc`, `zstd`, `zlib` (compression for the Zarr datasets)
* `nlohmann-json` (parses the JSON config sent on stdin by the Python side)
* `miniaudio` (WASAPI capture of the multichannel input device)
* `xtensor` with the `xsimd` feature, plus `xtl` (feature buffers and views)

vcpkg runs automatically through the preset (`VCPKG_MANIFEST_MODE` is ON). You only need vcpkg installed and reachable; the manifest installs the rest.

**Fetched at configure time (FetchContent, downloaded into the build tree):**
* `kfr` from git `main`. KFR 7 is required. It is built for AVX2 (`KFR_ARCH=avx2`) with single precision (`KFR_BASETYPE_F32`). If your CPU lacks AVX2, change `KFR_ARCH` (or enable multiarch) per KFR's docs, otherwise the binary will crash on an illegal instruction rather than fail to build.
* `z5` from git `master`, built with `BUILD_Z5PY=OFF` (we use z5 from C++ only; the Python side uses z5py separately).

**Install yourself (not managed by vcpkg or FetchContent):**
* LibTorch. The version here targets the CUDA 13 nightly debug build. Drop it next to the project or set `LIBTORCH_PATH` so `find_package(Torch)` resolves it.
* NVIDIA CUDA Toolkit 12.8 or newer, matched to your Visual Studio if you want IDE integration.
* NVIDIA cuDNN (its bin directory is bundled into the inference package on install).
* Ninja (the preset generator) and a Clang/clangd toolchain.

GPU architecture: `CMakeLists.txt` sets `TORCH_CUDA_ARCH_LIST "12.0"` (Blackwell, sm_120). On any other GPU, set this and `CMAKE_CUDA_ARCHITECTURES` to your compute capability or the inference build will not run on your card.

## Build targets and packages

There are two executables, and they have different dependency footprints. This matters because the data generation pipeline only needs one of them.

* `accdoa_gen` installs to `out/install/<preset>/Generate_Package/bin/accdoa_gen.exe`. It is compiled **without** Torch (`ACCDOA_NO_TORCH` is defined for its interface library), so it only needs the compression DLLs (blosc, zstd, zlib) alongside it. This is the binary the `audiogen-beamngpy` generation loop calls; that repo ships a prebuilt copy under `bin/`.
* `accdoa_infer` installs to `out/install/<preset>/Infer_Package/bin/accdoa_infer.exe`. It is compiled **with** Torch, and the install step copies the LibTorch and cuDNN runtime libraries into the same folder. Build this only when you need live inference.

So a generation-only setup never has to get LibTorch working; only the inference binary does.