#include <neo/fft.hpp>
#include <neo/math.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstdio>
#include <iostream>
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
auto as_mdspan_impl(py::array_t<T, Flags> buf, auto callback)
{
    auto const extents = stdex::dextents<size_t, Dim>{make_extents<Dim>(buf)};

    auto const map_right = stdex::layout_right::mapping<stdex::dextents<size_t, Dim>>{extents};
    if (is_layout_right(buf) and strides_match(buf, map_right)) {
        callback(stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_right>{buf.mutable_data(), map_right});
        return;
    }

    auto const map_left = stdex::layout_left::mapping<stdex::dextents<size_t, Dim>>{extents};
    if (is_layout_left(buf) and strides_match(buf, map_left)) {
        callback(stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_left>{buf.mutable_data(), map_left});
        return;
    }

    auto const strides = make_strides<Dim>(buf);
    auto const map     = stdex::layout_stride::mapping<stdex::dextents<size_t, Dim>>{extents, strides};
    callback(stdex::mdspan<T, stdex::dextents<size_t, Dim>, stdex::layout_stride>{buf.mutable_data(), map});
}

template<typename T, int Flags>
auto as_mdspan(py::array_t<T, Flags> buf, auto callback)
{
    switch (buf.ndim()) {
        case 1: as_mdspan_impl<1>(buf, callback); return;
        case 2: as_mdspan_impl<2>(buf, callback); return;
        case 3: as_mdspan_impl<3>(buf, callback); return;
        case 4: as_mdspan_impl<4>(buf, callback); return;
        case 5: as_mdspan_impl<5>(buf, callback); return;
        case 6: as_mdspan_impl<6>(buf, callback); return;
        case 7: as_mdspan_impl<7>(buf, callback); return;
        case 8: as_mdspan_impl<8>(buf, callback); return;
        default: throw std::runtime_error("unsupported ndim: " + std::to_string(buf.ndim()));
    }
}

template<typename T>
auto convolve(py::array_t<T, 0> array) -> void
{
    std::cout << typeid(T).name() << '\n';

    as_mdspan(array, []<typename Obj>(Obj obj) {
        if constexpr (std::same_as<typename Obj::layout_type, stdex::layout_left>) {
            std::printf("layout_left_%zud\n", obj.rank());
        }
        if constexpr (std::same_as<typename Obj::layout_type, stdex::layout_right>) {
            std::printf("layout_right_%zud\n", obj.rank());
        }
        if constexpr (std::same_as<typename Obj::layout_type, stdex::layout_stride>) {
            std::printf("layout_stride_%zud\n", obj.rank());
        }

        std::printf("size: %zu\n", obj.size());
    });
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = R"pbdoc(
        neo-sonar dsp library
        -----------------------

        .. currentmodule:: neo_dsp

        .. autosummary::
           :toctree: _generate

           a_weighting
           convolve
           fast_log2
           fast_log10
    )pbdoc";

    m.def("fast_log2", py::vectorize(neo::fast_log2));
    m.def("fast_log10", py::vectorize(neo::fast_log10));
    m.def("a_weighting", py::vectorize(neo::a_weighting<float>));
    m.def("a_weighting", py::vectorize(neo::a_weighting<double>));

    m.def("convolve", &convolve<float>);
    m.def("convolve", &convolve<double>);
    m.def("convolve", &convolve<std::complex<float>>);
    m.def("convolve", &convolve<std::complex<double>>);

#ifdef VERSION_INFO
    m.attr("__version__") = NEO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
