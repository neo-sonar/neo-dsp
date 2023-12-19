cmake_minimum_required(VERSION 3.15...3.27)
project(neo_dsp LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(mdspan GIT_REPOSITORY "https://github.com/kokkos/mdspan" GIT_TAG "stable" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(mdspan)

add_subdirectory(../src ${CMAKE_BINARY_DIR}/neo_dsp)

pybind11_add_module(_neo_dsp src/main.cpp)
target_link_libraries(_neo_dsp PRIVATE pybind11::headers neo::dsp)
target_compile_definitions(_neo_dsp PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _neo_dsp LIBRARY DESTINATION neo_dsp)
