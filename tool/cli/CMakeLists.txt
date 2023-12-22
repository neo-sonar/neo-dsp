project(neo_dsp_tools)

FetchContent_Declare(dr_libs GIT_REPOSITORY "https://github.com/mackron/dr_libs" GIT_TAG "master" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(dr_libs)

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
