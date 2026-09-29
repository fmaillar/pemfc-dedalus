from scripts.run_v07_n2_crossover_sensitivity import (
    monotonic_non_decreasing,
    monotonic_non_increasing,
)


def test_monotonic_helpers_accept_expected_trends():
    assert monotonic_non_decreasing([0.0, 0.1, 0.2])
    assert monotonic_non_increasing([1.0, 0.9, 0.8])


def test_monotonic_helpers_reject_reversals():
    assert not monotonic_non_decreasing([0.0, 0.2, 0.1])
    assert not monotonic_non_increasing([1.0, 0.8, 0.9])
