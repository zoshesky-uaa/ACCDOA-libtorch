cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(ACCDOA-libtorch LANGUAGES CXX)
enable_language(CUDA)

# Torch setup
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")

# Blackwell
unset(CMAKE_CUDA_ARCHITECTURES)
unset(CMAKE_CUDA_ARCHITECTURES CACHE)
set(TORCH_CUDA_ARCH_LIST "12.0")
set(CAFFE2_USE_CUDNN ON CACHE BOOL "" FORCE)
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

add_library(accdoa_core INTERFACE)

target_include_directories(accdoa_core INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${MINIAUDIO_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include
)

target_link_libraries(accdoa_core INTERFACE 
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

target_compile_definitions(accdoa_core INTERFACE 
    NOMINMAX             
    _USE_MATH_DEFINES
    KFR_BASETYPE_F32    
    CMK_AVX2         
    _HAS_CXX20=1       # ADD THIS: Forces MSVC to expose all C++20 
)

target_compile_options(accdoa_core INTERFACE 
    /utf-8
    -mavx2                       
    /bigobj                      
    -Wno-deprecated-declarations 
    /std:c++20
    /EHsc
)

# The Executables
add_executable(accdoa_gen 
    src/gen_main.cpp 
    src/generate.h   
)
target_link_libraries(accdoa_gen PRIVATE accdoa_core)

add_executable(accdoa_train 
    src/train_main.cpp 
    src/train.h     
)
target_link_libraries(accdoa_train PRIVATE accdoa_core)

add_executable(accdoa_infer 
    src/infer_main.cpp 
    src/inference.h  
)
target_link_libraries(accdoa_infer PRIVATE accdoa_core)

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")

foreach(tgt accdoa_gen accdoa_train accdoa_infer)
    set_property(TARGET ${tgt} PROPERTY COMPILE_FLAGS "${TORCH_CXX_FLAGS}")
    
    add_custom_command(TARGET ${tgt} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:${tgt}>
    )
endforeach()