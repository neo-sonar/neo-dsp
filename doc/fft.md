# FFT

## Interface

```cpp
namespace neo::fft {
    enum struct order : std::size_t
    {
    };

    enum struct direction : int
    {
        forward  = -1,
        backward = 1,
    };

    template<neo::complex Complex>
    struct fft_plan
    {
        using value_type = Complex;
        using size_type  = std::size_t;

        explicit fft_plan(fft::order order);

        [[nodiscard]] static constexpr auto max_order() noexcept -> fft::order;
        [[nodiscard]] static constexpr auto max_size() noexcept -> size_type;

        [[nodiscard]] auto order() const noexcept -> fft::order;
        [[nodiscard]] auto size() const noexcept -> size_type;

        template<inout_vector Vec>
            requires std::same_as<typename Vec::value_type, Complex>
        auto operator()(Vec x, direction dir) -> void;

        template<in_vector_of<Complex> InVec, out_vector_of<Complex> OutVec>
        auto operator()(InVec in, OutVec out, direction dir) -> void;
    };

    template<typename Plan, inout_vector Vec>
    constexpr auto fft(Plan& plan, Vec x) -> void;

    template<typename Plan, in_vector InVec, out_vector OutVec>
    constexpr auto fft(Plan& plan, InVec in, OutVec out) -> void;

    template<typename Plan, inout_vector Vec>
    constexpr auto ifft(Plan& plan, Vec x) -> void;

    template<typename Plan, in_vector InVec, out_vector OutVec>
    constexpr auto ifft(Plan& plan, InVec in, OutVec out) -> void;
}
```

## Resources

- [COMPUTING THE FAST FOURIER TRANSFORMON SIMD MICROPROCESSOR](https://www.cs.waikato.ac.nz/~ihw/PhD_theses/Anthony_Blake.pdf)
- [Notes on FFTs: for implementers](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/)
- [OTFFT Library](http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham1.html)
- [Discrete fourier transform and convolution -- scaling factor?](https://mathematica.stackexchange.com/questions/206788/discrete-fourier-transform-and-convolution-scaling-factor)
- [FXT Library](https://www.jjj.de/fxt/demo/arith/index.html)
- [FXT Library: Algorithms for programmers ideas and source code](http://dsp-book.narod.ru/fxtbook.pdf)

### DCT

- [Fast Cosine Transform via FFT](https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft)
- [Fast discrete cosine transform algorithms](https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms)

### DSP

- [musicinformationretrieval.com](https://musicinformationretrieval.com/index.html)
- [Audio Signal Processing for Machine Learning](https://youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&si=51JQNk_IuZSZITxX)
- [AudioLabsErlangen](https://www.youtube.com/@AudioLabsErlangen/videos)
- [CMUL via FMA](https://stackoverflow.com/questions/30089859/using-fma-fused-multiply-instructions-for-complex-multiplication)
