"""Fit the 3B material isotherms to the published MOF-303 and Al-fumarate water isotherms.

Source: Hanikel et al., ACS Cent. Sci. 5, 1699 (2019), doi:10.1021/acscentsci.9b00745, Fig. 2
(water isotherms at 30 C). The open-access figure is fetched at run time from Europe PMC
(PMC6813556 supplementary archive); it is never stored in this repository.

Method, fixed before any fit was looked at for the study:
  1. digitise: marker pixels of each material's colour, axes calibrated from the tick marks
     (x = 0 to 1 at pixels 102 to 632; y = 0 to 600 cm3/g at pixels 457 to 6); median uptake
     per 0.005 bin of P/Psat (adsorption and desorption branches are not separated);
  2. fit the three-term isotherm used by 3B (all terms share one heat of adsorption, so q* is
     a function of the single variable x = b(T) c and the spreading pressure stays closed form):
        q = q1 (RH/a1)/(1+RH/a1) + q2 (RH/a2)^n/(1+(RH/a2)^n) + q3 (RH/a3)/(1+RH/a3)   at 303 K;
  3. 90 % bootstrap intervals over bins (300 resamples, seed 2);
  4. a test of the temperature model the fit does not see: the constant-heat prediction of
     the step position at 40 and 60 C for Al-fumarate, against Laurenz et al. (2020), who
     report 0.25 at 30 C, 0.28 at 40 C and 0.33 at 60 C for a coated sample.

Writes verify/materials/fit_isotherms.json. Uses only numpy, scipy and Pillow.
"""
import io
import json
import os
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6813556/supplementaryFiles"
FIG = "oc9b00745_0002.jpg"
R_GAS = 8.314


def fetch_figure():
    with urllib.request.urlopen(URL, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    return np.asarray(Image.open(io.BytesIO(z.read(FIG))).convert("RGB")).astype(float)


def digitise(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    H, W, _ = im.shape
    rr, cc = np.mgrid[0:H, 0:W]
    interior = (cc > 104) & (cc < 655) & (rr > 8) & (rr < 455) & ~((rr > 300) & (cc > 300))
    masks = {"MOF-303": interior & (B > 180) & (G > 130) & (R < 130) & (B - R > 90),
             "Al-fumarate": interior & (R > 130) & (B > 170) & (G < 140) & (R - G > 40)}
    out = {}
    for name, m in masks.items():
        r, c = np.where(m)
        x = (c - 102) / 530.0
        y = (457 - r) / 75.17 * 100 / 22.414          # cm3(STP)/g -> mol/kg
        X, Y = [], []
        for lo in np.arange(0, 0.97, 0.005):
            s = (x >= lo) & (x < lo + 0.005)
            if s.sum() >= 3:
                X.append(lo + 0.0025)
                Y.append(np.median(y[s]))
        out[name] = (np.array(X), np.array(Y))
    return out


def model(p, rh):
    q1, a1, q2, a2, n, q3, a3 = p
    return q1 * (rh / a1) / (1 + rh / a1) + q2 * (rh / a2) ** n / (1 + (rh / a2) ** n) \
        + q3 * (rh / a3) / (1 + rh / a3)


def p_sat(T):  # Magnus, as in Paper 1
    Tc = T - 273.15
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


def step_rh(T, rh_30, dH):
    """Step position in RH at T for a constant heat dH, given its position at 303.15 K."""
    T0 = 303.15
    c_ratio = np.exp(-dH / R_GAS * (1 / T0 - 1 / T))  # c_step(T)/c_step(T0) = b(T0)/b(T), b = b0 exp(-dH/RT)
    return rh_30 * c_ratio * (T / T0) * p_sat(T0) / p_sat(T)


def step_shift_heats():
    """Concentration-based heat (what the dilute-solute model needs) and pressure-based isosteric
    heat, from the measured step positions at several temperatures.

    MOF-303: humidity at which the adsorption branch crosses 250, 300 and 350 cm3/g, from the
    digitised SI Fig. S17 (data/hanikel2019_mof303_figS17_digitised.csv, 10 to 50 C).
    Al-fumarate: step positions 0.25, 0.28, 0.33 at 30, 40, 60 C (Laurenz et al. 2020, coating).
    """
    import csv
    out = {}
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "hanikel2019_mof303_figS17_digitised.csv"))))
    pts = {}
    for r in rows:
        if r["branch"] == "ads":
            pts.setdefault(float(r["T_C"]), []).append((float(r["x"]), float(r["q"])))
    levels = {}
    for level in (250.0, 300.0, 350.0):
        TK, rh = [], []
        for T, v in sorted(pts.items()):
            v = np.array(sorted(v))
            x, q = v[:, 0], v[:, 1]
            i = np.where(q >= level)[0][0]
            rh.append(float(np.interp(level, [q[i - 1], q[i]], [x[i - 1], x[i]])))
            TK.append(T + 273.15)
        levels[f"{level:.0f}_cm3_g"] = _heats(np.array(TK), np.array(rh))
    out["MOF-303"] = levels
    out["Al-fumarate"] = _heats(np.array([303.15, 313.15, 333.15]), np.array([0.25, 0.28, 0.33]))
    return out


def _heats(TK, rh):
    c = rh * p_sat(TK) / (R_GAS * TK)
    p = rh * p_sat(TK)
    sc = np.polyfit(1 / TK, np.log(c), 1)[0]
    sp = np.polyfit(1 / TK, np.log(p), 1)[0]
    return {"T_K": TK.tolist(), "step_RH": rh.tolist(),
            "c_based_heat_kJ_mol": float(-R_GAS * sc / 1000),
            "pressure_based_Qst_kJ_mol": float(-R_GAS * sp / 1000)}


def main():
    data = digitise(fetch_figure())
    names = ["q1", "a1", "q2", "a2_step_RH", "n", "q3", "a3"]
    lb = [0, 1e-4, 1, 0.03, 1, 0, 0.05]
    ub = [15, 0.5, 40, 0.5, 40, 30, 20]
    res = {}
    for name, (X, Y) in data.items():
        p0 = [3, 0.02, 17, 0.13 if name == "MOF-303" else 0.256, 10, 8, 1.0]
        f = least_squares(lambda p: model(p, X) - Y, p0, bounds=(lb, ub))
        rng = np.random.default_rng(2)
        bs = []
        for _ in range(300):
            i = rng.integers(0, len(X), len(X))
            g = least_squares(lambda p: model(p, X[i]) - Y[i], f.x, bounds=(lb, ub))
            bs.append(g.x)
        bs = np.array(bs)
        rmse = float(np.sqrt(np.mean(f.fun ** 2)))
        res[name] = {"points": int(len(X)), "rmse_mol_kg": rmse, "rmse_rel_to_max": rmse / float(Y.max()),
                     **{k: float(v) for k, v in zip(names, f.x)},
                     "ci90": {k: [float(np.percentile(bs[:, j], 5)), float(np.percentile(bs[:, j], 95))]
                              for j, k in enumerate(names)},
                     "uptake_RH0.9_mol_kg": float(model(f.x, np.array([0.9]))[0]),
                     "uptake_RH0.10_mol_kg": float(model(f.x, np.array([0.10]))[0])}
    # temperature test for Al-fumarate (not used in the fit)
    rh30 = res["Al-fumarate"]["a2_step_RH"]
    res["temperature_test_Al-fumarate"] = {
        "dH_J_mol": -52200.0,
        "step_RH_30C_fit": rh30,
        "predicted_step_RH": {"40C": float(step_rh(313.15, rh30, -52200.0)),
                              "60C": float(step_rh(333.15, rh30, -52200.0))},
        "Laurenz_2020_reported": {"30C": 0.25, "40C": 0.28, "60C": 0.33},
        "note": "Laurenz et al. measured a binder-based coating; Hanikel et al. a powder."}
    # effective heats from the measured temperature shift of the step
    res["step_shift_heats"] = step_shift_heats()
    txt = json.dumps(res, indent=1)
    print(txt)
    with open(os.path.join(HERE, "fit_isotherms.json"), "w") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
