project(neo_convolution)

add_library(neo_convolution INTERFACE)
add_library(neo::convolution ALIAS neo_convolution)

target_include_directories(neo_convolution INTERFACE ${PROJECT_SOURCE_DIR})
target_link_libraries(neo_convolution INTERFACE neo::fft std::mdspan)
