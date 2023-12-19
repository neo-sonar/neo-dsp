cmake_minimum_required(VERSION 3.23...3.27)
project(neo_dsp LANGUAGES CXX VERSION ${CMAKE_PROJECT_VERSION})

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_neo_dsp src/main.cpp)
target_link_libraries(_neo_dsp PRIVATE pybind11::headers neo::dsp)
target_compile_definitions(_neo_dsp PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS _neo_dsp LIBRARY DESTINATION neo_dsp)
