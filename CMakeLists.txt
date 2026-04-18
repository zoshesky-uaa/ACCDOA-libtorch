# CMakeList.txt : CMake project for ACCDOA-libtorch, include source and define
# project specific logic here.
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
project (ACCDOA-libtorch LANGUAGES CXX CUDA)

add_definitions(-DKFR_NO_CUDA_EXTENSIONS)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Torch setup
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
find_package(Torch REQUIRED)

# Have to force avx2 because Ryzen 7000 CPU
set(KFR_ARCH "avx2" CACHE STRING "" FORCE)
set(KFR_ENABLE_MULTIARCH OFF CACHE BOOL "" FORCE)

include(FetchContent)
# xtensor and dependencies
FetchContent_Declare(xtl GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git GIT_TAG master OVERRIDE_FIND_PACKAGE)
FetchContent_Declare(xsimd GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git GIT_TAG master OVERRIDE_FIND_PACKAGE)
FetchContent_Declare(xtensor GIT_REPOSITORY https://github.com/xtensor-stack/xtensor GIT_TAG master OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(xtl xsimd xtensor)

# z5 dependencies
FetchContent_Declare(nlohmann_json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG master OVERRIDE_FIND_PACKAGE)
FetchContent_Declare(zlib GIT_REPOSITORY https://github.com/madler/zlib.git GIT_TAG master OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(nlohmann_json zlib)

# kfr
FetchContent_Declare(kfr GIT_REPOSITORY https://github.com/kfrlib/kfr.git GIT_TAG main)
FetchContent_MakeAvailable(kfr)

# miniaudio
FetchContent_Declare(miniaudio GIT_REPOSITORY https://github.com/mackron/miniaudio.git GIT_TAG master)
FetchContent_MakeAvailable(miniaudio)

# z5
set(BUILD_Z5PY OFF CACHE BOOL "" FORCE)
FetchContent_Declare(z5 GIT_REPOSITORY https://github.com/constantinpape/z5 GIT_TAG master)
FetchContent_MakeAvailable(z5)

add_executable(ACCDOA-libtorch 
    ACCDOA-libtorch.cpp)

set_target_properties(ACCDOA-libtorch PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

set_property(TARGET ACCDOA-libtorch PROPERTY COMPILE_FLAGS "${TORCH_CXX_FLAGS}")

target_link_libraries(ACCDOA-libtorch PRIVATE 
    xtl
    xsimd
    xtensor
    nlohmann_json
    zlib
    kfr
    miniaudio  
    z5
    ${TORCH_LIBRARIES}
)

target_include_directories(ACCDOA-libtorch PRIVATE
    ${zlib_SOURCE_DIR}
    ${zlib_BINARY_DIR}   
)

if (MSVC)
    target_compile_definitions(ACCDOA-libtorch PRIVATE 
        NOMINMAX             # Kills the min/max macros globally
        _USE_MATH_DEFINES    # Ensures M_PI is available
        KFR_BASETYPE_F32
    )

    target_compile_options(ACCDOA-libtorch PRIVATE 
        /arch:AVX2        # Tell MSVC it's allowed to understand KFR's AVX2 intrinsics
        /Zc:__cplusplus   # Some force for MSVC to set __cplusplus to the correct value
        /utf-8               # Fixes the Character Encoding warning (C4819)
        /bigobj              # Handles complex template bloat from Torch/KFR
    )

     # Copy LibTorch DLLs to the output directory so the .exe can find them at runtime
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ACCDOA-libtorch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:ACCDOA-libtorch>
    )

    # Enable Hot Reload (Edit and Continue)
    if (POLICY CMP0141)
        cmake_policy(SET CMP0141 NEW)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
    endif()
endif()