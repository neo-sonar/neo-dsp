project(neo_tools)

FetchContent_Declare(dr_libs GIT_REPOSITORY "https://github.com/mackron/dr_libs" GIT_TAG "master" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(dr_libs)

add_executable(neo_convolver)
target_sources(neo_convolver
    PRIVATE
        "convolver.cpp"
        "wav.cpp"
        "wav.hpp"
)

target_include_directories(neo_convolver
    PRIVATE
        ${dr_libs_SOURCE_DIR}
)

target_link_libraries(neo_convolver
    PRIVATE
        neosonar::neo
        fmt::fmt-header-only
)
