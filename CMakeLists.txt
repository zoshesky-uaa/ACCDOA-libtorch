# CMakeList.txt : CMake project for ACCDOA-libtorch, include source and define
# project specific logic here.
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
set(CMAKE_CXX_COMPILER "clang-cl" CACHE STRING "" FORCE)  
set(CMAKE_C_COMPILER   "clang-cl" CACHE STRING "" FORCE)
project (ACCDOA-libtorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Torch setup
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")

# Blackwell
unset(CMAKE_CUDA_ARCHITECTURES)
unset(CMAKE_CUDA_ARCHITECTURES CACHE)
set(TORCH_CUDA_ARCH_LIST "12.0")
find_package(Torch REQUIRED)

# VCPKG dependencies
find_package(ZLIB REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(blosc CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)
find_package(xtl CONFIG REQUIRED)
find_path(MINIAUDIO_INCLUDE_DIRS "miniaudio.h")

set(WITHIN_TRAVIS ON CACHE BOOL "" FORCE)
set(_VCPKG_TRIPLET_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")

# Feed z5's FindBLOSC.cmake search vars
set(BLOSC_SEARCH_HEADER_PATHS "${_VCPKG_TRIPLET_DIR}/include" CACHE STRING "" FORCE)
set(BLOSC_SEARCH_LIB_PATH     "${_VCPKG_TRIPLET_DIR}/lib"     CACHE STRING "" FORCE)

# Also hard-pin resolved values to bypass stale NOTFOUND cache
set(BLOSC_INCLUDE_DIR "${_VCPKG_TRIPLET_DIR}/include" CACHE PATH "" FORCE)
set(BLOSC_LIBRARIES   "${_VCPKG_TRIPLET_DIR}/lib/blosc.lib" CACHE FILEPATH "" FORCE)

include(FetchContent)
# kfr, need kfr 7
set(KFR_ARCH "avx2" CACHE STRING "" FORCE)
set(KFR_ENABLE_MULTIARCH OFF CACHE BOOL "" FORCE)
add_compile_definitions(KFR_BASETYPE_F32) 
FetchContent_Declare(kfr GIT_REPOSITORY https://github.com/kfrlib/kfr.git GIT_TAG main)
FetchContent_MakeAvailable(kfr)

# z5 FetchContext
set(BUILD_Z5PY OFF CACHE BOOL "" FORCE)
set(WITH_BLOSC ON CACHE BOOL "" FORCE)   
set(WITH_ZSTD ON CACHE BOOL "" FORCE) 
FetchContent_Declare(z5 GIT_REPOSITORY https://github.com/constantinpape/z5 GIT_TAG master)
FetchContent_MakeAvailable(z5)

add_executable(ACCDOA-libtorch ACCDOA-libtorch.cpp)

set_property(TARGET ACCDOA-libtorch PROPERTY COMPILE_FLAGS "${TORCH_CXX_FLAGS}")

target_link_libraries(ACCDOA-libtorch PRIVATE 
    xtl    
    xtensor::use_xsimd
    ZLIB::ZLIB
    kfr 
    kfr_io 
    kfr_dft 
    kfr_dsp
    kfr_audio
    blosc_shared
    zstd::libzstd
    nlohmann_json::nlohmann_json
    z5
    ${TORCH_LIBRARIES}
)

target_include_directories(ACCDOA-libtorch PRIVATE ${MINIAUDIO_INCLUDE_DIRS})

target_compile_definitions(ACCDOA-libtorch PRIVATE 
    NOMINMAX             
    _USE_MATH_DEFINES
    KFR_BASETYPE_F32    
    CMK_AVX2           
)

target_compile_options(ACCDOA-libtorch PRIVATE 
    /utf-8
    -mavx2                       # Force AVX2
    /bigobj                      # Handles template bloat from Torch/xtensor
    -Wno-deprecated-declarations # Suppresses noisy warnings
)

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