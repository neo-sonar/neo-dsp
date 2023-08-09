#include "wav.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>

namespace neo {

namespace {

template<typename To, typename From>
    requires((sizeof(To) == sizeof(From)) && std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>)
auto bitCast(From const& src) noexcept -> To
{
#if defined(__cpp_lib_bit_cast)
    return std::bit_cast<To>(src);
#elif __has_builtin(__builtin_bit_cast) or defined(_MSC_VER)
    return __builtin_bit_cast(To, src);
#else
    // This implementation additionally requires destination type to be
    // trivially constructible
    static_assert(is_trivially_constructible_v<To>);
    To dst{};
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
#endif
}

template<typename T>
[[nodiscard]] auto from_little_endian(std::span<std::byte const, sizeof(T)> src) -> T
{
    auto bytes = std::array<std::byte, sizeof(T)>{};
    std::copy(begin(src), end(src), std::begin(bytes));

    if constexpr (std::endian::native == std::endian::big) {
        std::reverse(bytes.begin(), bytes.end());
        return bitCast<T>(bytes);
    }

    return bitCast<T>(bytes);
}

struct span_input_stream
{
    std::span<std::byte const> buf;
};

auto skip(span_input_stream& stream, std::size_t count) -> void { stream.buf = stream.buf.subspan(count); }

auto read(span_input_stream& stream, std::span<std::byte> out) -> void
{
    std::copy(begin(stream.buf), std::next(begin(stream.buf), ssize(out)), begin(out));
    stream.buf = stream.buf.subspan(size(out));
}

template<size_t Size>
auto read_bytes(span_input_stream& stream) -> std::array<std::byte, Size>
{
    auto buffer = std::array<std::byte, Size>{};
    read(stream, buffer);
    return buffer;
}

template<typename T>
auto read_little_endian(span_input_stream& stream) -> T
{
    auto const buffer = read_bytes<sizeof(T)>(stream);
    return from_little_endian<T>(buffer);
}

auto s16_to_f32(std::span<std::byte const> buf) -> float
{
    auto const scale = 0.000030517578125F;
    auto const word  = from_little_endian<std::int16_t>(buf.first<2>());
    return static_cast<float>(word) * scale;
}

auto s24_to_f32(std::span<std::byte const> buf) -> float
{
    auto const scale = 0.00000011920928955078125;
    auto const a     = static_cast<std::uint32_t>(buf[0]) << 8;
    auto const b     = static_cast<std::uint32_t>(buf[1]) << 16;
    auto const c     = static_cast<std::uint32_t>(buf[2]) << 24;
    auto const x     = static_cast<std::int32_t>(a | b | c) >> 8;
    return static_cast<float>(static_cast<double>(x) * scale);
}

auto f32_to_f32(std::span<std::byte const> buf) -> float { return from_little_endian<float>(buf.first<4>()); }

auto f64_to_f32(std::span<std::byte const> buf) -> float
{
    return static_cast<float>(from_little_endian<double>(buf.first<8>()));
}

}  // namespace

auto parse_wav_header(std::span<std::byte const> data) -> wav_file
{
    static constexpr auto junk_id = std::array{
        std::byte{'J'},
        std::byte{'U'},
        std::byte{'N'},
        std::byte{'K'},
    };

    auto stream = span_input_stream{data};
    auto wav    = wav_file{};

    wav.header.magic_header = read_bytes<4>(stream);
    wav.header.file_size    = read_little_endian<std::uint32_t>(stream);
    wav.header.format       = read_bytes<4>(stream);
    wav.header.subchunk_id  = read_bytes<4>(stream);

    if (wav.header.subchunk_id == junk_id) {
        skip(stream, read_little_endian<std::uint32_t>(stream));
        wav.header.subchunk_id = read_bytes<4>(stream);
    }

    wav.header.subchunk_size   = read_little_endian<std::uint32_t>(stream);
    wav.header.format_tag      = read_little_endian<wav_format_tag>(stream);
    wav.header.channels        = read_little_endian<std::uint16_t>(stream);
    wav.header.sample_rate     = read_little_endian<std::uint32_t>(stream);
    wav.header.byte_rate       = read_little_endian<std::uint32_t>(stream);
    wav.header.block_align     = read_little_endian<std::uint16_t>(stream);
    wav.header.bits_per_sample = read_little_endian<std::uint16_t>(stream);
    wav.header.data_id         = read_bytes<4>(stream);
    wav.header.data_size       = read_little_endian<std::uint32_t>(stream);

    wav.data = stream.buf;
    return wav;
}

wav_decoder::wav_decoder(wav_file const& file) : _file{file}
{
    if (_file.header.format_tag == wav_format_tag::pcm) {
        if (_file.header.bits_per_sample == 16) { _decoder = s16_to_f32; }
        if (_file.header.bits_per_sample == 24) { _decoder = s24_to_f32; }
    }

    if (_file.header.format_tag == wav_format_tag::ieee_float) {
        if (_file.header.bits_per_sample == 32) { _decoder = f32_to_f32; }
        if (_file.header.bits_per_sample == 64) { _decoder = f64_to_f32; }
    }
}

auto wav_decoder::operator()(std::span<float> out) -> std::span<float>
{
    if (_decoder == nullptr) { throw std::runtime_error{"decoder unimplemented"}; }

    auto const numChannels    = _file.header.channels;
    auto const bytesPerSample = _file.header.bits_per_sample / 8;

    auto const totalNumFrames = _file.header.data_size / _file.header.block_align;
    auto const frameRemaining = static_cast<size_t>(totalNumFrames - _frameIndex);
    if (_frameIndex >= totalNumFrames) { return {}; }

    auto const numFrames = std::min(out.size() / numChannels, frameRemaining);
    for (auto f{_frameIndex}; f < _frameIndex + numFrames; ++f) {
        auto const fIndex = f * _file.header.block_align;
        for (auto ch{0U}; ch < numChannels; ++ch) {
            auto const cIndex                         = fIndex + ch * bytesPerSample;
            auto const sample                         = _decoder(_file.data.subspan(cIndex, bytesPerSample));
            out[(f - _frameIndex) * numChannels + ch] = sample;
        }
    }

    _frameIndex += numFrames;
    return out.subspan(0, numFrames * numChannels);
}

}  // namespace neo
