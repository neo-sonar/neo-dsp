project(perceptual-convolution VERSION ${CMAKE_PROJECT_VERSION})

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
        PluginEditor.hpp
        PluginEditor.cpp
        PluginParameter.hpp
        PluginParameter.cpp
        PluginProcessor.hpp
        PluginProcessor.cpp

        dsp/convolution.cpp
        dsp/convolution.hpp
        dsp/normalize.cpp
        dsp/normalize.hpp
        dsp/render.hpp
        dsp/resample.cpp
        dsp/resample.hpp
        dsp/spectogram.cpp
        dsp/spectogram.hpp
        dsp/stft.cpp
        dsp/stft.hpp
        dsp/wav.cpp
        dsp/wav.hpp
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
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/3rd_party/JUCE/modules
)

target_link_libraries(PerceptualConvolution
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp

    PUBLIC
        neo::convolution
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
