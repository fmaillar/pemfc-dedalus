from scripts.run_v07_anode_volume_sensitivity import (
    DEFAULT_VOLUMES_M3,
    first_time_to_fraction,
)


def test_default_volume_sensitivity_values_are_log_spaced_around_reference():
    assert DEFAULT_VOLUMES_M3 == (
        5.0e-6,
        10.0e-6,
        20.0e-6,
        40.0e-6,
        80.0e-6,
    )


def test_first_time_to_fraction_returns_first_crossing():
    rows = [
        {"time_s": 0.0, "anode_relative_humidity": 0.0},
        {"time_s": 10.0, "anode_relative_humidity": 0.2},
        {"time_s": 20.0, "anode_relative_humidity": 0.5},
        {"time_s": 30.0, "anode_relative_humidity": 0.8},
    ]
    assert first_time_to_fraction(rows, equilibrium_rh=1.0, fraction=0.5) == 20.0
    assert first_time_to_fraction(rows, equilibrium_rh=1.0, fraction=0.9) is None
