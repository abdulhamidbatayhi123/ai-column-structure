"""S's metriplectic term is scaled by the time scale and by the pilot's typical entropy-gradient size
sqrt(1e3 kappa) (the median |P grad S^| that kappa is computed from), so S's vector field does not
depend on the units of entropy. Without it the term carried |P grad S^| (about 40 J/K per latent unit
at 128 cells), moved z by several latent units per step after Adam's first update, and 6 of 10 training
iterations of S were skipped at 128 cells, 200 snapshots, d = 16 (S-M, without the term: none)."""
import torch

from p3b.latent.arms_structured import SDynamics

F64 = torch.float64
D, K = 4, 3


class _LF:
    """A latent-functional stub whose entropy is S0 + c * (a quadratic) in latent space."""
    system = "A"

    def __init__(self, c):
        self.c = c
        g = torch.Generator().manual_seed(0)
        self.wE = torch.randn(D, dtype=F64, generator=g)
        self.wN = torch.randn(D, dtype=F64, generator=g)
        self.Q = torch.randn(D, D, dtype=F64, generator=g)

    def evaluate(self, z, raw, grads=True, create_graph=False):
        n = z.shape[0]
        dS = self.c * (z @ (self.Q + self.Q.T))
        return {"E": z @ self.wE, "N": z @ self.wN, "S": self.c * (z @ self.Q * z).sum(-1),
                "dE": self.wE.expand(n, D), "dN": self.wN.expand(n, D), "dS": dS,
                "P_E": torch.full((n,), 0.3, dtype=F64), "P_N": torch.full((n,), -0.1, dtype=F64),
                "Phi_S": self.c * torch.full((n,), 0.05, dtype=F64)}


def _field(c, metriplectic=True):
    torch.manual_seed(1)
    kappa = 1e-3 * 2.0 * c ** 2                 # kappa = 1e-3 x median |P grad S|^2 scales with c^2
    s = SDynamics(D, K, _LF(c), kappa=kappa, width=16, metriplectic=metriplectic, time_scale=100.0)
    with torch.no_grad():                        # a non-trivial A, as after training
        for p in (s.A_net.parameters() if s.A_net is not None else []):
            p.copy_(0.3 * torch.randn_like(p))
    z = torch.randn(5, D, dtype=F64)
    feat = torch.zeros(5, K, dtype=F64)
    return s(z, feat, None)[0]


def test_s_vector_field_does_not_depend_on_the_units_of_entropy():
    a, b = _field(1.0), _field(1000.0)
    assert torch.allclose(a, b, rtol=1e-9, atol=1e-12)


def test_metriplectic_term_is_order_one_over_the_time_scale_for_unit_weights():
    full = _field(40.0)
    port_only = _field(40.0, metriplectic=False)
    term = (full - port_only).norm(dim=-1)
    assert float(term.max()) < 100.0 / 100.0             # |A|^2 ~ O(10) here, so at most ~10 / time scale
