cmake_minimum_required(VERSION 3.16)
project(ncnn_ocr VERSION 1.0.0 LANGUAGES CXX C)

# ============================================================
# C++ Standard
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Build options
# ============================================================
option(USE_VULKAN "Enable Vulkan GPU acceleration" OFF)
option(USE_OPENMP "Enable OpenMP parallelism" ON)

# ============================================================
# Find ncnn
# ============================================================
# User can set ncnn_DIR to the ncnn install path
if(NOT DEFINED ncnn_DIR)
    # Auto-detect relative to this CMakeLists (src/): <root>/../ncnn/build/install/lib/cmake/ncnn
    # src/ -> .. = ncnn_ocr -> .. = pnnx -> ncnn
    get_filename_component(_ncnn_root "${CMAKE_CURRENT_SOURCE_DIR}/../../ncnn/build/install/lib/cmake/ncnn" ABSOLUTE)
    if(EXISTS "${_ncnn_root}/ncnnConfig.cmake")
        set(ncnn_DIR "${_ncnn_root}")
    else()
        # Fallback: try system installation
        find_path(ncnn_INCLUDE_DIR ncnn/net.h)
        find_library(ncnn_LIBRARY ncnn)
        if(ncnn_INCLUDE_DIR AND ncnn_LIBRARY)
            add_library(ncnn STATIC IMPORTED)
            set_target_properties(ncnn PROPERTIES
                IMPORTED_LOCATION "${ncnn_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ncnn_INCLUDE_DIR}"
            )
        endif()
    endif()
endif()

if(DEFINED ncnn_DIR)
    find_package(ncnn REQUIRED)
endif()

if(NOT TARGET ncnn)
    message(FATAL_ERROR "ncnn not found. Set ncnn_DIR to the ncnn install path.\n"
        "Example: cmake -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn ..")
endif()

# ============================================================
# Dependencies (all local, no downloads needed)
# ============================================================

# nlohmann/json — header-only, local copy under <root>/3rd/
set(JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rd/nlohmannjson/single_include")
if(NOT EXISTS "${JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "nlohmann/json not found at ${JSON_INCLUDE_DIR}")
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE "${JSON_INCLUDE_DIR}")

# stb_image — header-only, copied from ncnn_llm
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/stb_image.h")
    message(FATAL_ERROR "stb_image.h not found in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# ============================================================
# OpenMP (optional)
# ============================================================
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
    else()
        message(STATUS "OpenMP not found, will use ncnn's built-in thread pool")
    endif()
endif()

# ============================================================
# Vulkan (optional)
# ============================================================
if(USE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found")
        add_definitions(-DNCNN_VULKAN=1)
    else()
        message(WARNING "Vulkan not found, disabling GPU acceleration")
        set(USE_VULKAN OFF)
    endif()
endif()

# ============================================================
# Multi-OCR library (HunYuan + GLM)
# ============================================================
add_library(ocr_core STATIC
    bpe_tokenizer.h
    bpe_tokenizer.cpp
    hunyuan_ocr.h
    hunyuan_ocr.cpp
    glm_ocr.h
    glm_ocr.cpp
    paddleocr_vl.h
    paddleocr_vl.cpp
    expression_layer.h
    expression_layer.cpp
    image_utils.h
    image_utils.cpp
    rope_embed.h
    rope_embed.cpp
    vision_rope.h
    vision_rope.cpp
    sampling.h
    sampling.cpp
    text_runtime.h
    text_runtime.cpp
)

target_include_directories(ocr_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ocr_core PUBLIC
    ncnn
    nlohmann_json
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ocr_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(WIN32)
    # Windows-specific: disable min/max macros
    target_compile_definitions(ocr_core PUBLIC NOMINMAX)
endif()

# ============================================================
# Main executable
# ============================================================
add_executable(ncnn_ocr main.cpp)
target_link_libraries(ncnn_ocr PRIVATE ocr_core)

install(TARGETS ncnn_ocr RUNTIME DESTINATION bin)

# ============================================================
# Platform-specific settings
# ============================================================
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_options(ocr_core PRIVATE /W3 /utf-8)
    target_compile_options(ncnn_ocr PRIVATE /utf-8)
else()
    target_compile_options(ocr_core PRIVATE -Wall -Wextra)
endif()

