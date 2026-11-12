"""Revision R3 groundwork: system B's fluid-solid exchange coefficient as a function of the flow,
h_v(|v|), used by every copy of the store's right-hand side. The default (no model) is the constant
h_v of the registered design, so nothing changes until the model is set from the primary source."""
import numpy as np
import torch

from p3b.solvers.c_plus_l import TorchStoreRHS
from p3b.thermal_store import ThermalStore


def _state(st):
    z = (np.arange(st.N) + 0.5) / st.N
    Tf = 300 + 40 / (1 + np.exp((z - 0.4) / 0.05))
    Ts = Tf - 3 * np.exp(-((z - 0.4) / 0.1) ** 2)
    return np.concatenate([st.Cf * (Tf - 293.15), st.Cs * (Ts - 293.15)])


def test_without_a_model_h_v_is_the_registered_constant():
    st = ThermalStore(16, d_p=0.05, compensate=True)
    assert st.h_v_of(1e-3) == st.h_v_of(-5e-3) == st.h_v == 2.0e4


def test_a_model_sets_the_exchange_in_the_reference_rhs():
    model = lambda speed: 1.0e4 + 3.0e6 * speed            # noqa: E731  (any non-negative function of |v|)
    a = ThermalStore(16, d_p=0.05, compensate=True)
    b = ThermalStore(16, d_p=0.05, compensate=True, h_v_model=model)
    y = _state(a)
    for v in (2e-3, -4e-3):
        assert b.h_v_of(v) == model(abs(v))
        fa, fb = a.fluxes(y, (v, 353.15, 293.15)), b.fluxes(y, (v, 353.15, 293.15))
        assert np.allclose(fb["W"], fa["W"] * model(abs(v)) / a.h_v, rtol=1e-14)


def test_the_torch_rhs_follows_the_same_model():
    model = lambda speed: 1.0e4 + 3.0e6 * speed            # noqa: E731
    st = ThermalStore(16, d_p=0.05, compensate=True, h_v_model=model)
    y = _state(st)
    for v in (2e-3, -4e-3):
        inp = (v, 353.15, 293.15)
        ref = st.rhs(0.0, y, inp)
        got = TorchStoreRHS(st)(torch.tensor(y, dtype=torch.float64), torch.tensor(inp, dtype=torch.float64))
        assert np.allclose(got.numpy(), ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_the_model_is_recorded_with_the_store_and_absent_without_one():
    def wakao(speed):
        return 2.0e4
    assert ThermalStore(16, d_p=0.05, h_v_model=wakao).params["h_v_model"] == "wakao"
    assert "h_v_model" not in ThermalStore(16, d_p=0.05).params    # older data's constants stay valid


def test_the_muscl_store_keeps_the_exchange_model():
    from p3b.solvers.coarse import MusclStore

    def wakao(speed):
        return 1.0e4 + 3.0e6 * speed
    st = ThermalStore(16, d_p=0.05, compensate=True, h_v_model=wakao)
    m = MusclStore.from_store(st)
    assert m.h_v_of(2e-3) == st.h_v_of(2e-3) and not m.compensate
