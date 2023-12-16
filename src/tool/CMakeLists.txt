project(neo_dsp_tools)

add_executable(neo_dsp_convolver)
target_sources(neo_dsp_convolver
    PRIVATE
        "convolver.cpp"
        "wav.cpp"
        "wav.hpp"
)

target_include_directories(neo_dsp_convolver
    PRIVATE
        ${dr_libs_SOURCE_DIR}
)

target_link_libraries(neo_dsp_convolver
    PRIVATE
        neo::dsp
        fmt::fmt-header-only
)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(neo_dsp_convolver PRIVATE ${ACCELERATE_LIBRARY})
endif()
