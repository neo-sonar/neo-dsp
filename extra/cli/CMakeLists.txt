project(neosonar-neo-cli)

FetchContent_Declare(dr_libs GIT_REPOSITORY "https://github.com/mackron/dr_libs" GIT_TAG "master" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(dr_libs)

add_executable(neo-cli)
target_sources(neo-cli
    PRIVATE
        "src/convolver.cpp"
        "src/wav.cpp"
        "src/wav.hpp"
)

target_include_directories(neo-cli
    PRIVATE
        ${dr_libs_SOURCE_DIR}
)

target_link_libraries(neo-cli
    PRIVATE
        neosonar::neo
        fmt::fmt-header-only
)
