cmake_minimum_required(VERSION 3.23...3.27)
project(neo-dsp VERSION ${CMAKE_PROJECT_VERSION})

add_library(neo-dsp INTERFACE)
add_library(neo::dsp ALIAS neo-dsp)

target_link_libraries(neo-dsp INTERFACE std::mdspan)
target_compile_features(neo-dsp INTERFACE cxx_std_20)
target_include_directories(neo-dsp INTERFACE ${PROJECT_SOURCE_DIR})

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(neo-dsp INTERFACE ${ACCELERATE_LIBRARY})
endif()
