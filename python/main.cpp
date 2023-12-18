#include <neo/algorithm.hpp>
#include <neo/complex.hpp>
#include <neo/fft.hpp>
#include <neo/math.hpp>
#include <neo/unit.hpp>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <bit>
#include <complex>
#include <cstdio>
#include <iostream>
#include <optional>
#include <typeindex>

namespace py = pybind11;

template<typename T, int Flags>
[[nodiscard]] auto is_layout_left(py::array_t<T, Flags> buf) -> bool
{
    return (buf.flags() & py::array::f_style) != 0;
}

template<typename T, int Flags>
[[nodiscard]] auto is_layout_right(py::array_t<T, Flags> buf) -> bool
{
    return (buf.flags() & py::array::c_style) != 0;
}

template<typename T, int Flags, typename Mapping>
[[nodiscard]] auto strides_match(py::array_t<T, Flags> buf, Mapping map) -> bool
{
    for (auto i{0U}; i < map.extents().rank(); ++i) {
        if (map.stride(i) != static_cast<size_t>(buf.strides(i) / buf.itemsize())) {
            return false;
        }
    }
    return true;
}

template<size_t Dim, typename T, int Flags>
[[nodiscard]] auto make_extents(py::array_t<T, Flags> buf) -> std::array<size_t, Dim>
{
    auto ext = std::array<size_t, Dim>{};
    for (auto i{0U}; i < Dim; ++i) {
        ext[i] = static_cast<size_t>(buf.shape(i));
    }
    return ext;
}

template<size_t Dim, typename T, int Flags>
[[nodiscard]] auto make_strides(py::array_t<T, Flags> buf) -> std::array<size_t, Dim>
{
    auto strides = std::array<size_t, Dim>{};
    for (auto i{0U}; i < Dim; ++i) {
        strides[i] = static_cast<size_t>(buf.strides(i) / buf.itemsize());
    }
    return strides;
}

template<std::size_t Dim, typename T, int Flags>
auto to_mdspan_layout_right(py::array_t<T, Flags> buf)
{
    auto const extents = stdex::dextents<size_t, Dim>{make_extents<Dim>(buf)};
    auto const mapping = stdex::layout_right::mapping<stdex::dextents<size_t, Dim>>{extents};
    return stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_right>{
        buf.mutable_data(),
        mapping,
    };
}

template<std::size_t Dim, typename T, int Flags>
auto to_mdspan_layout_stride(py::array_t<T, Flags> buf)
{
    auto const strides = make_strides<Dim>(buf);
    auto const extents = stdex::dextents<size_t, Dim>{make_extents<Dim>(buf)};
    auto const mapping = stdex::layout_stride::mapping<stdex::dextents<size_t, Dim>>{extents, strides};
    return stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_stride>{
        buf.mutable_data(),
        mapping,
    };
}

template<std::size_t Dim, typename T, int Flags>
auto as_mdspan_impl(py::array_t<T, Flags> buf, auto func)
{
    auto const extents = stdex::dextents<size_t, Dim>{make_extents<Dim>(buf)};

    auto const mapping_right = stdex::layout_right::mapping<stdex::dextents<size_t, Dim>>{extents};
    if (is_layout_right(buf) and strides_match(buf, mapping_right)) {
        return func(stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_right>{
            buf.mutable_data(),
            mapping_right,
        });
    }

    auto const mapping_left = stdex::layout_left::mapping<stdex::dextents<size_t, Dim>>{extents};
    if (is_layout_left(buf) and strides_match(buf, mapping_left)) {
        return func(stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_left>{
            buf.mutable_data(),
            mapping_left,
        });
    }

    return func(to_mdspan_layout_stride<Dim>(buf));
}

template<typename T, int Flags>
auto as_mdspan(py::array_t<T, Flags> buf, auto func)
{
    switch (buf.ndim()) {
        case 1: return as_mdspan_impl<1>(buf, func);
        case 2: return as_mdspan_impl<2>(buf, func);
        case 3: return as_mdspan_impl<3>(buf, func);
        default: throw std::runtime_error("unsupported ndim: " + std::to_string(buf.ndim()));
    }
}

template<neo::complex Complex, neo::fft::direction Dir>
auto fft(py::array_t<Complex> array, std::optional<std::size_t> n, neo::fft::norm norm) -> py::array_t<Complex>
{
    using Float = typename Complex::value_type;

    return as_mdspan(array, [n, norm]<typename Vec>(Vec input) -> py::array_t<Complex> {
        if constexpr (Vec::rank() == 1) {
            auto const size  = n.value_or(input.extent(0));
            auto const order = neo::ilog2(size);
            if (not std::has_single_bit(size)) {
                throw std::runtime_error{"unsupported size: " + std::to_string(size)};
            }

            auto plan   = neo::fft::fft_plan<Complex>{order, Dir};
            auto result = py::array_t<Complex>(static_cast<py::ssize_t>(size));
            auto out    = to_mdspan_layout_right<1>(result);
            if constexpr (Dir == neo::fft::direction::forward) {
                neo::fft::fft(plan, input, out);
                if (norm == neo::fft::norm::forward) {
                    neo::scale(Float(1) / Float(size), out);
                }
            } else {
                neo::fft::ifft(plan, input, out);
                if (norm == neo::fft::norm::backward) {
                    neo::scale(Float(1) / Float(size), out);
                }
            }

            if (norm == neo::fft::norm::ortho) {
                neo::scale(Float(1) / std::sqrt(Float(size)), out);
            }

            return result;
        } else {
            throw std::runtime_error{"unsupported rank: " + std::to_string(input.rank())};
        }
    });
}

template<std::floating_point Float>
[[nodiscard]] auto to_decibels(Float amplitude) -> Float
{
    return neo::amplitude_to_db<neo::precision::accurate, Float>(amplitude);
}

template<std::floating_point Float>
[[nodiscard]] auto to_decibels_estimate(Float amplitude) -> Float
{
    return neo::amplitude_to_db<neo::precision::estimate, Float>(amplitude);
}

[[nodiscard]] auto rfftfreq(std::size_t n, double invSampleRate) -> py::array_t<double>
{
    auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
    neo::rfftfreq(to_mdspan_layout_right<1>(out), invSampleRate);
    return out;
}

PYBIND11_MODULE(_neo_dsp, m)
{
    py::enum_<neo::fft::norm>(m, "norm")
        .value("backward", neo::fft::norm::backward)
        .value("ortho", neo::fft::norm::ortho)
        .value("forward", neo::fft::norm::forward);

    m.def("rfftfreq", &rfftfreq);

    m.def("fft", &fft<std::complex<float>, neo::fft::direction::forward>);
    m.def("fft", &fft<std::complex<double>, neo::fft::direction::forward>);

    m.def("ifft", &fft<std::complex<float>, neo::fft::direction::backward>);
    m.def("ifft", &fft<std::complex<double>, neo::fft::direction::backward>);

    m.def("amplitude_to_db", py::vectorize(to_decibels<float>));
    m.def("amplitude_to_db", py::vectorize(to_decibels<double>));

    m.def("amplitude_to_db_estimate", py::vectorize(to_decibels_estimate<float>));
    m.def("amplitude_to_db_estimate", py::vectorize(to_decibels_estimate<double>));

    m.def("a_weighting", py::vectorize(neo::a_weighting<float>));
    m.def("a_weighting", py::vectorize(neo::a_weighting<double>));

    m.def("fast_log2", py::vectorize(neo::fast_log2));
    m.def("fast_log10", py::vectorize(neo::fast_log10));

#ifdef VERSION_INFO
    m.attr("__version__") = NEO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
