cmake_minimum_required(VERSION 3.15...3.27)
project(neo-dsp VERSION ${CMAKE_PROJECT_VERSION})

add_library(neo_dsp INTERFACE)
add_library(neo::dsp ALIAS neo_dsp)

target_compile_definitions(neo_dsp INTERFACE _CRT_SECURE_NO_WARNINGS=1)
target_include_directories(neo_dsp INTERFACE ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_dsp INTERFACE std::mdspan)
