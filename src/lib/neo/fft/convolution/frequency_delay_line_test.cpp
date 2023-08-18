#include "frequency_delay_line.hpp"

#include <catch2/catch_template_test_macros.hpp>

TEMPLATE_TEST_CASE("neo/fft/convolution: fdl_index", "", int, unsigned, std::ptrdiff_t, std::size_t)
{
    using Index = TestType;

    auto indexer = neo::fft::fdl_index<Index>{3};

    SECTION("copy callback")
    {
        auto ignore = [](auto, auto) {};

        indexer([](auto i) { REQUIRE(i == Index(0)); }, ignore);
        indexer([](auto i) { REQUIRE(i == Index(1)); }, ignore);
        indexer([](auto i) { REQUIRE(i == Index(2)); }, ignore);
        indexer([](auto i) { REQUIRE(i == Index(0)); }, ignore);

        indexer.reset();
        indexer([](auto i) { REQUIRE(i == Index(0)); }, ignore);
        indexer([](auto i) { REQUIRE(i == Index(1)); }, ignore);
        indexer([](auto i) { REQUIRE(i == Index(2)); }, ignore);
    }

    SECTION("multiply callback")
    {
        auto check_multiply_iteration_1 = [loop_count = 0](auto fdl, auto filter) mutable {
            if (loop_count == 0) {
                REQUIRE(fdl == Index(0));
                REQUIRE(filter == Index(0));
            }
            if (loop_count == 1) {
                REQUIRE(fdl == Index(1));
                REQUIRE(filter == Index(2));
            }
            if (loop_count == 2) {
                REQUIRE(fdl == Index(2));
                REQUIRE(filter == Index(1));
            }

            ++loop_count;
        };

        auto check_multiply_iteration_2 = [loop_count = 0](auto fdl, auto filter) mutable {
            if (loop_count == 0) {
                REQUIRE(fdl == Index(0));
                REQUIRE(filter == Index(1));
            }
            if (loop_count == 1) {
                REQUIRE(fdl == Index(1));
                REQUIRE(filter == Index(0));
            }
            if (loop_count == 2) {
                REQUIRE(fdl == Index(2));
                REQUIRE(filter == Index(2));
            }

            ++loop_count;
        };

        indexer.reset();
        indexer([](auto i) { REQUIRE(i == Index(0)); }, check_multiply_iteration_1);
        indexer([](auto i) { REQUIRE(i == Index(1)); }, check_multiply_iteration_2);
    }
}
