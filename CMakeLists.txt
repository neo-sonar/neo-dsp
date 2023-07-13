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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
enable_testing()

include(FetchContent)
FetchContent_Declare(mdspan_repo GIT_REPOSITORY "https://github.com/kokkos/mdspan" GIT_TAG "stable")
FetchContent_MakeAvailable(mdspan_repo)

add_subdirectory(3rd_party/JUCE)
set_directory_properties(PROPERTIES JUCE_COMPANY_COPYRIGHT "Copyright 2023 neo-sonar. All rights reserved.")
set_directory_properties(PROPERTIES JUCE_COMPANY_NAME "neo-sonar")
set_directory_properties(PROPERTIES JUCE_VST3_COPY_DIR "${CMAKE_BINARY_DIR}/VST3")
set_directory_properties(PROPERTIES JUCE_AU_COPY_DIR "${CMAKE_BINARY_DIR}/AU")

add_subdirectory(3rd_party/neo-fft/include)
add_subdirectory(src/lib)
add_subdirectory(src/plugin)
add_subdirectory(src/tests)
