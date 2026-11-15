"""Revision R3: the fluid transport properties behind h(v) and k_z(v), computed once and recorded.

Each system's model has constant fluid properties, so its transport properties are taken at the one
state those constants describe:
  * system A, air at 293.15 K and 101325 Pa: Paper 1's rho_g = 1.2 kg/m3 (C_pg = 1000 J/kg/K) is air
    near 293 K;
  * system B, liquid water at 318.15 K and 101325 Pa: the store's rho_f = 990 kg/m3 and
    c_f = 4180 J/kg/K are water near 318 K.

Air: the pseudo-pure-fluid equation of state of Lemmon et al. (2000) with the viscosity and thermal
conductivity of Lemmon and Jacobsen (2004). Water: IAPWS-95, the IAPWS 2008 viscosity (Huber et al.
2009) and the IAPWS 2011 thermal conductivity (Huber et al. 2012). Both are evaluated with CoolProp
(Bell et al. 2014). The bibliography keys CoolProp reports for each model are written to the output.

Run (CoolProp is not a project dependency; install it anywhere and point COOLPROP_DIR at it):
    pip install --target <dir> CoolProp==8.0.0
    COOLPROP_DIR=<dir> python verify/r3/properties.py
Writes verify/r3/properties.json.
"""
import json
import os
import sys

if os.environ.get("COOLPROP_DIR"):
    sys.path.insert(0, os.environ["COOLPROP_DIR"])
import CoolProp  # noqa: E402
from CoolProp.CoolProp import PropsSI, get_BibTeXKey  # noqa: E402

P = 101325.0
STATES = {"air_A": ("Air", 293.15), "water_B": ("Water", 318.15)}


def state(fluid: str, T: float) -> dict:
    rho = PropsSI("D", "T", T, "P", P, fluid)
    cp = PropsSI("C", "T", T, "P", P, fluid)
    mu = PropsSI("V", "T", T, "P", P, fluid)
    k = PropsSI("L", "T", T, "P", P, fluid)
    return {"fluid": fluid, "T_K": T, "P_Pa": P, "rho_kg_m3": rho, "cp_J_kgK": cp, "mu_Pa_s": mu,
            "k_W_mK": k, "Pr": cp * mu / k,
            "bibtex": {kind: get_BibTeXKey(fluid, kind) for kind in ("EOS", "VISCOSITY", "CONDUCTIVITY")}}


def main() -> dict:
    out = {"coolprop_version": CoolProp.__version__, "states": {n: state(f, T) for n, (f, T) in STATES.items()}}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "properties.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
