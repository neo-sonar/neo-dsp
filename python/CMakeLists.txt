cmake_minimum_required(VERSION 3.15...3.26)
project(neo_dsp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
    mdspan
    GIT_REPOSITORY "https://github.com/kokkos/mdspan"
    GIT_TAG "stable"
    GIT_PROGRESS TRUE
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(mdspan)

add_subdirectory(../src/lib ${CMAKE_BINARY_DIR}/neo_fft)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(neo_dsp main.cpp)
target_compile_definitions(neo_dsp PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(neo_dsp PRIVATE neo::fft)
install(TARGETS neo_dsp LIBRARY DESTINATION .)
