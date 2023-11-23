cmake_minimum_required(VERSION 3.23)
project(perceptual-convolution-dev VERSION 0.1.0)

find_program(CCACHE ccache)
if (CCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif ()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()
include(CTest)
include(FetchContent)

FetchContent_Declare(
    JUCE
    GIT_REPOSITORY "https://github.com/juce-framework/JUCE"
    GIT_TAG "develop"
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    mdspan
    GIT_REPOSITORY "https://github.com/kokkos/mdspan"
    GIT_TAG "stable"
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    dr_libs
    GIT_REPOSITORY "https://github.com/mackron/dr_libs.git"
    GIT_TAG "master"
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY "https://github.com/catchorg/Catch2.git"
    GIT_TAG devel
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(JUCE)
FetchContent_MakeAvailable(mdspan)
FetchContent_MakeAvailable(dr_libs)
FetchContent_MakeAvailable(Catch2)
include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

add_subdirectory(src/lib)
add_subdirectory(src/benchmarks)
add_subdirectory(src/plugin)
add_subdirectory(src/tool)
