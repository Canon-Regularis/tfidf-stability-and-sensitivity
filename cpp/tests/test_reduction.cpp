// Reduction policies.
//
// The policies must differ in the right way and agree in the right way. If they
// never differed, the noise-floor measurement that derives tau (section 7.1)
// would be measuring nothing; if they disagreed on exactly representable
// inputs, one of them would be wrong.
#include <tfidf/core/reduction.hpp>

#include <doctest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace tfidf;

namespace {

/// Bit-level equality, which is what "bit-exact" means here. `==` would
/// conflate -0.0 with 0.0 and call two NaNs unequal.
bool same_bits(Real a, Real b) {
    return std::memcmp(&a, &b, sizeof(Real)) == 0;
}

Real naive_reference(const std::vector<Real>& v) {
    Real s = 0.0;
    for (const Real x : v) {
        s += x;
    }
    return s;
}

}  // namespace

TEST_CASE("reduce: every policy agrees on exactly representable values") {
    // Powers of two sum exactly, so rounding cannot distinguish the policies.
    const std::vector<Real> v{1.0, 2.0, 4.0, 8.0, 0.5, 0.25};
    for (const auto p : {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                         Reduction::Exact}) {
        CHECK(reduce::sum(v, p) == 15.75);
    }
}

TEST_CASE("reduce: the empty and singleton sums") {
    const std::vector<Real> empty{};
    const std::vector<Real> one{42.5};
    for (const auto p : {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                         Reduction::Exact}) {
        CHECK(reduce::sum(empty, p) == 0.0);
        CHECK(reduce::sum(one, p) == 42.5);
    }
}

TEST_CASE("reduce: Naive is exactly a left-to-right fold") {
    // The normative policy, so it must match the literal reading of the formula
    // and nothing cleverer.
    std::mt19937_64 rng(20260811);
    std::uniform_real_distribution<Real> mag(-1.0, 1.0);
    std::uniform_int_distribution<int> exp10(-12, 3);
    std::vector<Real> v;
    v.reserve(5000);
    for (int i = 0; i < 5000; ++i) {
        v.push_back(mag(rng) * std::pow(10.0, exp10(rng)));
    }
    CHECK(same_bits(reduce::sum(v, Reduction::Naive), naive_reference(v)));
}

TEST_CASE("reduce: compensation recovers what the naive fold discards") {
    // Each addend is individually below half an ulp of the running total, so
    // the naive fold loses all of them; their exact sum is not.
    std::vector<Real> v{1.0};
    v.insert(v.end(), 100, 1e-17);
    REQUIRE(1.0 + 1e-17 == 1.0);

    const Real naive = reduce::sum(v, Reduction::Naive);
    const Real exact = reduce::sum(v, Reduction::Exact);
    const Real neumaier = reduce::sum(v, Reduction::Neumaier);

    CHECK(naive == 1.0);
    CHECK(exact > naive);
    CHECK(neumaier > naive);
    // Exact is correctly rounded, hence the ground truth for the others.
    CHECK(std::abs(exact - (1.0 + 100 * 1e-17)) <= std::numeric_limits<Real>::epsilon());
}

TEST_CASE("reduce: Exact recovers a result naive arithmetic destroys") {
    // The classic ill-conditioned sum. Left to right:
    //   0 + 1e100      -> 1e100
    //   1e100 + 1.0    -> 1e100   (the first 1.0 is below half an ulp: lost)
    //   1e100 - 1e100  -> 0
    //   0 + 1.0        -> 1.0     (the second 1.0 survives, having arrived
    //                              after the cancellation)
    // so naive returns 1.0: one of the two units is lost. The exact sum is 2.0.
    const std::vector<Real> v{1e100, 1.0, -1e100, 1.0};
    CHECK(reduce::sum(v, Reduction::Exact) == 2.0);
    CHECK(reduce::sum(v, Reduction::Naive) == 1.0);
}

TEST_CASE("reduce: Exact reproduces the documented fsum half-even case") {
    // CPython's own regression case for the correction step in Exact::value():
    // the 1e-16 nudges the 1 closer to two, so a correctly-rounded collapse
    // must round the last digit up rather than down.
    const std::vector<Real> v{1e-16, 1.0, 1e16};
    CHECK(reduce::sum(v, Reduction::Exact) == 1.0000000000000002e16);
}

TEST_CASE("reduce: Exact is order-independent") {
    // A correctly-rounded sum depends only on the multiset, never on order,
    // which is what lets it serve as ground truth.
    std::mt19937_64 rng(7);
    std::vector<Real> v;
    for (int i = 0; i < 400; ++i) {
        v.push_back(std::ldexp(1.0, i % 80 - 40) * ((i % 3) ? 1.0 : -1.0));
    }
    const Real forward = reduce::sum(v, Reduction::Exact);
    std::shuffle(v.begin(), v.end(), rng);
    CHECK(same_bits(reduce::sum(v, Reduction::Exact), forward));
}

TEST_CASE("reduce: Naive is NOT order-independent") {
    // Counterpart of the previous test. If this became order independent, the
    // summation-order sensitivity the paper studies would have disappeared.
    std::vector<Real> v{1.0};
    v.insert(v.end(), 100, 1e-17);
    const Real small_first = [&] {
        std::vector<Real> w(v.rbegin(), v.rend());
        return reduce::sum(w, Reduction::Naive);
    }();
    CHECK(reduce::sum(v, Reduction::Naive) != small_first);
}

TEST_CASE("reduce: Pairwise beats Naive on a long uniform sum") {
    // 1e6 identical values: pairwise error grows as O(log n), naive as O(n).
    const std::vector<Real> v(1u << 20, 0.1);
    const Real exact = reduce::sum(v, Reduction::Exact);
    const Real naive_err = std::abs(reduce::sum(v, Reduction::Naive) - exact);
    const Real pair_err = std::abs(reduce::sum(v, Reduction::Pairwise) - exact);
    CHECK(pair_err <= naive_err);
}

TEST_CASE("reduce: overflow follows IEEE-754 and is not silently papered over") {
    // {1e308, 1e308, ...} overflows on the first addition. That is the correct
    // IEEE result, and README section 6 applies no stabilising transformation,
    // so `inf` is what the specification calls for. Asserted so a later rescaled
    // sum cannot change every published digit unnoticed.
    const std::vector<Real> overflowing{1e308, 1e308, -1e308, -1e308};
    CHECK(std::isinf(reduce::sum(overflowing, Reduction::Naive)));

    // Just below the overflow threshold everything stays finite and exact.
    const std::vector<Real> large{1e308, -1e308, 1.0};
    for (const auto p : {Reduction::Naive, Reduction::Neumaier, Reduction::Pairwise,
                         Reduction::Exact}) {
        const Real s = reduce::sum(large, p);
        CHECK(std::isfinite(s));
        CHECK(s == 1.0);
    }
}
