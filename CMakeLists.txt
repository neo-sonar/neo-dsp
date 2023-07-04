cmake_minimum_required(VERSION 3.23)
project(perceptual-convolution VERSION 0.1.0)

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

include(FetchContent)
FetchContent_Declare(mdspan_repo GIT_REPOSITORY "https://github.com/kokkos/mdspan" GIT_TAG "stable")
FetchContent_MakeAvailable(mdspan_repo)

add_subdirectory(3rd_party/JUCE)
set_directory_properties(PROPERTIES JUCE_COMPANY_COPYRIGHT "Copyright 2023 neo-sonar. All rights reserved.")
set_directory_properties(PROPERTIES JUCE_COMPANY_NAME "neo-sonar")
set_directory_properties(PROPERTIES JUCE_VST3_COPY_DIR "${CMAKE_BINARY_DIR}/VST3")
set_directory_properties(PROPERTIES JUCE_AU_COPY_DIR "${CMAKE_BINARY_DIR}/AU")

juce_add_plugin(PerceptualConvolution
    PRODUCT_NAME "Perceptual Convolution"            # The name of the final executable, which can differ from the target name
    BUNDLE_ID "com.neo-sonar.perceptual-convolution" # Should uniquely identify this target
    PLUGIN_CODE Pcnv                                 # A unique four-character plugin id with exactly one upper-case character
    PLUGIN_MANUFACTURER_CODE Neos                    # A four-character manufacturer id with at least one upper-case character
    FORMATS AU VST3 Standalone                       # The formats to build. Other valid formats are: AAX Unity VST AU AUv3
    HARDENED_RUNTIME_ENABLED TRUE                    # Enables macOS' hardened runtime for this target. Required for notarisation.
    COPY_PLUGIN_AFTER_BUILD TRUE                     # Should the plugin be installed to a default location after building?
    IS_SYNTH FALSE                                   # Is this a synth or an effect?
    NEEDS_MIDI_INPUT FALSE                           # Does the plugin need midi input?
    NEEDS_MIDI_OUTPUT FALSE                          # Does the plugin need midi output?
    IS_MIDI_EFFECT FALSE                             # Is this plugin a MIDI effect?
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE                # Does the editor need keyboard focus?
)

target_sources(PerceptualConvolution
    PRIVATE
        src/PluginEditor.hpp
        src/PluginEditor.cpp
        src/PluginParameter.hpp
        src/PluginParameter.cpp
        src/PluginProcessor.hpp
        src/PluginProcessor.cpp

        src/neo/fft.cpp
        src/neo/fft.hpp
        src/neo/math.cpp
        src/neo/math.hpp
        src/neo/mdspan.hpp
        src/neo/resample.cpp
        src/neo/resample.hpp
)

target_compile_definitions(PerceptualConvolution
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_USE_MP3AUDIOFORMAT=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
        JUCE_MODAL_LOOPS_PERMITTED=1
        JUCE_REPORT_APP_USAGE=0
        JUCE_DISABLE_JUCE_VERSION_PRINTING=1
)

target_include_directories(PerceptualConvolution
    PUBLIC
        src
        3rd_party/JUCE/modules
)

target_link_libraries(PerceptualConvolution
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp

    PUBLIC
        std::mdspan
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
