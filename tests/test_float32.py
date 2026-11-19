"""float32 rounding and ulp distance, for values produced outside the package."""

from __future__ import annotations

import math

import pytest

from tfidf_stability.utils.numerics import float32_round, ulps_between_f32

_F32_MAX = float.fromhex("0x1.fffffep+127")
_ONE_PLUS_ULP = 1.0 + 2.0**-23


def test_a_binary64_value_rounds_to_the_nearest_float32() -> None:
    assert float32_round(1.0) == 1.0
    assert float32_round(0.1) == float.fromhex("0x1.99999ap-4")
    assert float32_round(-0.1) == -float.fromhex("0x1.99999ap-4")


def test_a_halfway_value_rounds_to_even() -> None:
    assert float32_round(1.0 + 2.0**-24) == 1.0, "halfway to 1 + 2**-23: the even neighbour is 1"
    assert float32_round(1.0 + 3 * 2.0**-24) == 1.0 + 2.0**-22


def test_a_value_past_the_range_rounds_to_an_infinity_of_its_sign() -> None:
    assert float32_round(1e39) == math.inf
    assert float32_round(-1e39) == -math.inf
    assert float32_round(_F32_MAX + 2.0**103 - 2.0**80) == _F32_MAX, "inside half an ulp"


def test_infinities_nan_and_signed_zero_pass_through() -> None:
    assert float32_round(math.inf) == math.inf
    assert math.isnan(float32_round(math.nan))
    assert math.copysign(1.0, float32_round(-0.0)) == -1.0


def test_subnormals_round_on_the_float32_grid() -> None:
    tiny = 2.0**-149
    assert float32_round(tiny) == tiny
    assert float32_round(2.0**-150) == 0.0, "halfway to the smallest subnormal: even is zero"
    assert float32_round(0.75 * tiny) == tiny


def test_the_ulp_distance_is_in_float32_units() -> None:
    assert ulps_between_f32(1.0, _ONE_PLUS_ULP) == 1.0
    assert ulps_between_f32(_ONE_PLUS_ULP, 1.0) == -1.0, "signed as b - a"
    assert ulps_between_f32(0.0, 2.0**-149) == 1.0
    assert ulps_between_f32(_F32_MAX, float.fromhex("0x1.fffffcp+127")) == -1.0
    assert ulps_between_f32(2.0, 1.0) == -(2.0**23) / 2


def test_equal_values_and_signed_zeros_are_zero_apart() -> None:
    assert ulps_between_f32(0.5, 0.5) == 0.0
    assert ulps_between_f32(-0.0, 0.0) == 0.0


def test_a_non_finite_value_is_infinitely_far() -> None:
    assert ulps_between_f32(1.0, math.inf) == math.inf
    assert ulps_between_f32(math.nan, 1.0) == math.inf


@pytest.mark.parametrize(("a", "b"), [(0.1, 1.0), (1.0, 0.1)])
def test_a_value_float32_cannot_hold_is_refused(a: float, b: float) -> None:
    with pytest.raises(ValueError, match=r"0\.1 is not a float32 value"):
        ulps_between_f32(a, b)
