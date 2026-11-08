"""Multi-term isotherms that share one heat of adsorption (formulation, Section 3.2).

q*(c, T) = Q(x),  x = b(T) c,  b(T) = b0 exp(-dH / (R T)),
Q(x)  = sum_k q_k (x/a_k)^{n_k} / (1 + (x/a_k)^{n_k}),
Pi(x) = integral_0^x Q(x')/x' dx' = sum_k (q_k/n_k) ln(1 + (x/a_k)^{n_k})   (closed form).

Because every term is a function of the same x, the inverse isotherm is a one-dimensional
monotone root and the formulation's closed forms carry over unchanged. Paper 1's dual-site
isotherm is the two-term special case (`from_paper1`); the 3B materials are three-term fits
(`from_fit`, docs/materials_3B.md).
"""
import numpy as np

R = 8.314


def p_sat(T):  # Magnus, as in Paper 1
    Tc = T - 273.15
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


class MultiTermIsotherm:
    def __init__(self, terms, b0, dH):
        self.terms = [(float(q), float(a), float(n)) for q, a, n in terms]
        self.b0 = float(b0)
        self.dH = float(dH)

    def b(self, T):
        return self.b0 * np.exp(-self.dH / (R * T))

    def Q(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for q, a, n in self.terms:
            y = (x / a) ** n
            out = out + q * y / (1 + y)
        return out

    def Pi(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for q, a, n in self.terms:
            out = out + q / n * np.log1p((x / a) ** n)
        return out

    def X(self, q):
        """Inverse isotherm, by bisection on log x (robust, vectorised)."""
        q = np.asarray(q, dtype=float)
        lo = np.full_like(q, -60.0)
        hi = np.full_like(q, 60.0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = self.Q(np.exp(mid)) < q
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return np.exp(0.5 * (lo + hi))

    def q_star(self, c, T):
        return self.Q(self.b(T) * np.clip(c, 0.0, None))

    @property
    def q_capacity(self):
        return sum(q for q, _, _ in self.terms)


def from_paper1(phys):
    """Paper 1's dual-site isotherm as a two-term MultiTermIsotherm (x = b_H c)."""
    n = float(getattr(phys, "isotherm_n", 1.0))
    fH = float(getattr(phys, "henry_fraction", 1.0))
    r = phys.b0 / phys.b_H0
    return MultiTermIsotherm([(phys.q_max * fH, 1.0, 1.0), (phys.q_max * (1 - fH), 1.0 / r, n)],
                             b0=phys.b_H0, dH=phys.delta_H)


def from_fit(fit, dH, T_ref=303.15):
    """A three-term isotherm from verify/materials/fit_isotherms.json (fit in RH at T_ref).

    b0 is chosen so that x = RH at T_ref: b(T_ref) c_sat(T_ref) = 1.
    """
    c_sat = p_sat(T_ref) / (R * T_ref)
    b0 = np.exp(dH / (R * T_ref)) / c_sat
    terms = [(fit["q1"], fit["a1"], 1.0), (fit["q2"], fit["a2_step_RH"], fit["n"]), (fit["q3"], fit["a3"], 1.0)]
    return MultiTermIsotherm(terms, b0=b0, dH=dH)
