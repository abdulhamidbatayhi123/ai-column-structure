"""Materials and bed constants (plan 1, Task 2). Expected values: verify/timing_materials.json."""
import pytest

from p3b import materials as m


def test_mof303_feed_values():
    T, rh, v = 303.15, 0.40, 0.10
    iso = m.load_isotherm("MOF-303")
    phys = m.bed_physics(v, T)
    c_in = m.c_in_from_rh(rh, T)
    q = float(iso.q_star(c_in, T))
    assert q == pytest.approx(22.2003, abs=1e-3)
    assert m.k_ldf(iso, phys, c_in, T) == pytest.approx(4.9702e-4, abs=1e-7)
    assert m.t_stoich(iso, phys, c_in, T) / 3600 == pytest.approx(5.8745, abs=1e-3)


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        m.load_isotherm("ZIF-8")


def test_heats_are_the_measured_shift_values():
    assert m.DH_C == {"MOF-303": -51.3e3, "Al-fumarate": -48.5e3}
    assert m.load_isotherm("Al-fumarate").dH == -48.5e3
