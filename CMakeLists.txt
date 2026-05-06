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

# Common flags for all binaries
add_library(accdoa_common_flags INTERFACE)
target_include_directories(accdoa_common_flags INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(accdoa_common_flags INTERFACE 
    NOMINMAX             
    _USE_MATH_DEFINES
    KFR_BASETYPE_F32    
    CMK_AVX2         
    _HAS_CXX20=1       
)

target_compile_options(accdoa_common_flags INTERFACE 
    /utf-8
    -mavx2                 
    /bigobj                      
    -Wno-deprecated-declarations 
    /std:c++20
    /EHsc
)


# --- Generator Library (No Torch) ---
add_library(accdoa_lib_gen INTERFACE)
target_link_libraries(accdoa_lib_gen INTERFACE 
    accdoa_common_flags
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
)
target_include_directories(accdoa_lib_gen INTERFACE ${MINIAUDIO_INCLUDE_DIRS})
# Pass macro to drop Torch headers in generator build (requires matching #ifdef)
target_compile_definitions(accdoa_lib_gen INTERFACE ACCDOA_NO_TORCH)

# --- Training Library (No Miniaudio / KFR) ---
add_library(accdoa_lib_train INTERFACE)
target_link_libraries(accdoa_lib_train INTERFACE 
    accdoa_common_flags
    xtl    
    xtensor::use_xsimd
    ZLIB::ZLIB
    blosc_shared
    zstd::libzstd
    nlohmann_json::nlohmann_json
    z5
    ${TORCH_LIBRARIES}
)
target_include_directories(accdoa_lib_train INTERFACE 
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include
)

# --- Inference Library (Everything included) ---
add_library(accdoa_lib_infer INTERFACE)
target_link_libraries(accdoa_lib_infer INTERFACE 
    accdoa_common_flags
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
target_include_directories(accdoa_lib_infer INTERFACE 
    ${MINIAUDIO_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include
)

add_executable(accdoa_gen src/gen_main.cpp src/generate.h)
target_link_libraries(accdoa_gen PRIVATE accdoa_lib_gen)

add_executable(accdoa_train src/train_main.cpp src/train.h)
target_link_libraries(accdoa_train PRIVATE accdoa_lib_train)

add_executable(accdoa_infer src/infer_main.cpp src/inference.h)
target_link_libraries(accdoa_infer PRIVATE accdoa_lib_infer)

if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()

set_property(TARGET accdoa_gen   PROPERTY RUNTIME_DEPENDENCY_SET gen_deps)
set_property(TARGET accdoa_train PROPERTY RUNTIME_DEPENDENCY_SET train_deps)
set_property(TARGET accdoa_infer PROPERTY RUNTIME_DEPENDENCY_SET infer_deps)

# By adding the COMPONENT tag, CPack will allow you to generate distinct 
# package zips/installers for Generate, Train, and Inference standalone.

install(TARGETS accdoa_gen   RUNTIME_DEPENDENCY_SET gen_deps   RUNTIME DESTINATION Generate_Package/bin COMPONENT Generate_Package)
install(TARGETS accdoa_train RUNTIME_DEPENDENCY_SET train_deps RUNTIME DESTINATION Train_Package/bin    COMPONENT Train_Package)
install(TARGETS accdoa_infer RUNTIME_DEPENDENCY_SET infer_deps RUNTIME DESTINATION Infer_Package/bin    COMPONENT Infer_Package)

# Define regex patterns to filter out virtual API sets and system-level DLLs
set(API_SET_EXCLUDES "^(api|ext)-ms-.*")
set(SYSTEM_DIR_EXCLUDES "^[A-Za-z]:[\\\\/][Ww][Ii][Nn][Dd][Oo][Ww][Ss][\\\\/].*")

# Install the actual runtime dependencies (vcpkg DLLs, etc.) but ignore Windows system files
install(RUNTIME_DEPENDENCY_SET gen_deps   
    DESTINATION Generate_Package/bin COMPONENT Generate_Package
    PRE_EXCLUDE_REGEXES ${API_SET_EXCLUDES}
    POST_EXCLUDE_REGEXES ${SYSTEM_DIR_EXCLUDES}
)

install(RUNTIME_DEPENDENCY_SET train_deps 
    DESTINATION Train_Package/bin COMPONENT Train_Package
    DIRECTORIES "${TORCH_INSTALL_PREFIX}/lib" "${TORCH_INSTALL_PREFIX}/bin"
    PRE_EXCLUDE_REGEXES ${API_SET_EXCLUDES}
    POST_EXCLUDE_REGEXES ${SYSTEM_DIR_EXCLUDES}
)

install(RUNTIME_DEPENDENCY_SET infer_deps 
    DESTINATION Infer_Package/bin COMPONENT Infer_Package
    DIRECTORIES "${TORCH_INSTALL_PREFIX}/lib" "${TORCH_INSTALL_PREFIX}/bin" 
    PRE_EXCLUDE_REGEXES ${API_SET_EXCLUDES}
    POST_EXCLUDE_REGEXES ${SYSTEM_DIR_EXCLUDES}
)