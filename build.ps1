mkdir build
cd build
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PSScriptRoot\libtorch"
cmake --build build --config Debug