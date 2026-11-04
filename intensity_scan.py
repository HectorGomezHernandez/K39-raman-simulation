"""
Scan over multiple intensity values per beam at fixed waist (1 mm-equivalent
power-density semantics): for every (transition x polarization x geometry x
intensity) combination, compute the smallest Delta in [50, 4000] MHz where
P_sc(Delta) crosses 5% and 10%, plus the corresponding t_pi at those Delta.

Prints a small reference showing what (waist, power) combinations give
each scanned intensity (assuming the standard 50/50 power split between
the two Raman beams).

Output: intensity_scan_summary.txt next to the script.
"""
import numpy as np
import pandas as pd

from Raman_simulation import (
    D1, Beam, diagonalize_manifold, dipole_operator_uncoupled,
    transform_operator_to_eigenbasis, reduced_dipole_from_linewidth,
    select_states_by_mF, excited_manifold_reference_energy,
    excited_state_offsets, evaluate_parameter_set, C,
)


INTENSITIES_MWCM2 = [159.155, 500.0, 1000.0, 1500.0, 2000.0]
DELTA_GRID = np.linspace(50.0, 4000.0, 1000)


def precompute(line, B_G, mF_i, mF_f):
    gs, Eg, Ug, bg = diagonalize_manifold(
        J=0.5, I=C.I_nuclear, ahfs=C.ahfs_S, gJ=C.gJ_S, gI=C.gI,
        B_G=B_G, manifold_label="S")
    es, Ee, Ue, be = diagonalize_manifold(
        J=line.J_e, I=C.I_nuclear, ahfs=line.ahfs, gJ=line.gJ, gI=C.gI,
        B_G=B_G, manifold_label="P", bhfs=line.bhfs)
    d_red = reduced_dipole_from_linewidth(line.omega, line.gamma, Je=line.J_e)
    Dp = {q: transform_operator_to_eigenbasis(
            dipole_operator_uncoupled(be, bg, line.J_e, 0.5, q, d_red), Ue, Ug)
          for q in (-1, 0, +1)}
    gi = select_states_by_mF(gs, mF_i)[0][0]
    gf = select_states_by_mF(gs, mF_f)[0][0]
    mF_int = sorted({mF_i + q for q in (-1, 0, 1)} & {mF_f + q for q in (-1, 0, 1)})
    ei = [i for mf in mF_int for i, _ in select_states_by_mF(es, mf)]
    Eref = excited_manifold_reference_energy(Ee)
    dn = excited_state_offsets(Ee, ei, Eref)
    return Eg, Ee, Dp, gi, gf, ei, dn


def first_crossing(x, y, target):
    """First Delta where P_sc(Delta) crosses target, linearly interpolated.
    Returns None if never reached."""
    for k in range(len(x) - 1):
        y0, y1 = y[k], y[k + 1]
        if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
            t = (target - y0) / (y1 - y0)
            return float(x[k] + t * (x[k + 1] - x[k]))
    return None


def make_beam_pair(separation_deg, alpha1, phi1, alpha2, phi2):
    h = separation_deg / 2.0
    return (Beam(theta_deg=+h, alpha_deg=alpha1, phi_deg=phi1),
            Beam(theta_deg=-h, alpha_deg=alpha2, phi_deg=phi2))


transitions = [
    (402.7, 1.0, 0.0, "402.7 G, 1->0"),
    (445.4, 1.0, 0.0, "445.4 G, 1->0"),
    (526.0, -1.0, 0.0, "526 G, -1->0"),
    (526.0, 0.0, 1.0, "526 G, 0->1"),
]
polarizations = [
    ("lin45",     45, 0,  45, 0),
    ("pi+sigma",   0, 0,  90, 0),
    ("circular",  45, 90, 45, 90),
]
separations = [0.0, 66.4, 133.6, 180.0]


def intensity_to_beam_params(I_mWcm2):
    """For a given peak intensity per beam, return (P_mW @ w0=1mm, w0_mm @ P=5mW)."""
    P_mW_at_1mm = np.pi * I_mWcm2 / 100.0      # see derivation in main()
    w0_mm_at_5mW = np.sqrt(500.0 / (np.pi * I_mWcm2))
    return P_mW_at_1mm, w0_mm_at_5mW


def main():
    # --- Reference: I -> (P, w0) ---
    print("Intensity to (waist, power) lookup")
    print("=" * 70)
    print(f"{'I (mW/cm^2)':>12s}   {'P @ w0=1mm (mW)':>16s}   {'w0 @ P=5mW (mm)':>16s}")
    print("-" * 70)
    lookup_lines = []
    for I in INTENSITIES_MWCM2:
        P_mW, w0_mm = intensity_to_beam_params(I)
        line = f"{I:12.1f}   {P_mW:16.2f}   {w0_mm:16.3f}"
        print(line)
        lookup_lines.append(line)

    # --- Sweep ---
    rows = []
    for B_G, mFi, mFf, tlabel in transitions:
        Eg, Ee, Dp, gi, gf, ei, dn = precompute(D1, B_G, mFi, mFf)
        for plabel, a1, p1, a2, p2 in polarizations:
            for sep in separations:
                beam1, beam2 = make_beam_pair(sep, a1, p1, a2, p2)
                for I_mWcm2 in INTENSITIES_MWCM2:
                    psc = np.zeros_like(DELTA_GRID)
                    tpi = np.zeros_like(DELTA_GRID)
                    for k, D_MHz in enumerate(DELTA_GRID):
                        r = evaluate_parameter_set(
                            I_mWcm2, D_MHz, 0.0, gi, gf, ei,
                            Eg, Ee, dn, Dp, D1.gamma,
                            omega_laser=D1.omega, beam1=beam1, beam2=beam2,
                        )
                        psc[k] = r["P_sc_avg"]
                        tpi[k] = r["t_pi_us"]
                    D5  = first_crossing(DELTA_GRID, psc, 0.05)
                    D10 = first_crossing(DELTA_GRID, psc, 0.10)
                    # t_pi at those Delta (interpolated)
                    tpi_at = lambda D: float(np.interp(D, DELTA_GRID, tpi)) if D is not None else None
                    rows.append({
                        "transition": tlabel,
                        "pol": plabel,
                        "sep_deg": sep,
                        "I_mWcm2": I_mWcm2,
                        "Delta_5pct_MHz": D5,
                        "tpi_5pct_us":   tpi_at(D5),
                        "Delta_10pct_MHz": D10,
                        "tpi_10pct_us":  tpi_at(D10),
                    })
                    print(f"{tlabel:18s} | {plabel:8s} | sep={sep:5.1f} | "
                          f"I={I_mWcm2:7.1f} | "
                          f"D5={('---' if D5 is None else f'{D5:7.1f}')} "
                          f"(t_pi={('---' if D5 is None else f'{tpi_at(D5):.2f} us')})  "
                          f"D10={('---' if D10 is None else f'{D10:7.1f}')} "
                          f"(t_pi={('---' if D10 is None else f'{tpi_at(D10):.2f} us')})")

    df = pd.DataFrame(rows)

    # --- Save ---
    out = "intensity_scan_summary.txt"
    with open(out, "w") as f:
        f.write("Intensity scan: smallest positive Delta in [50, 4000] MHz where\n")
        f.write("P_sc(Delta) crosses 5% and 10%, plus the corresponding t_pi.\n")
        f.write("(Linear interpolation of P_sc(Delta); '---' = threshold not reached.)\n\n")
        f.write("Intensity to (waist, power) lookup:\n")
        f.write(f"{'I (mW/cm^2)':>12s}   {'P @ w0=1mm (mW)':>16s}   {'w0 @ P=5mW (mm)':>16s}\n")
        f.write("-" * 70 + "\n")
        for line in lookup_lines:
            f.write(line + "\n")
        f.write("\n" + "=" * 100 + "\n")
        f.write(df.to_string(index=False, float_format=lambda v: "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"))
        f.write("\n")
    df.to_csv("intensity_scan_summary.csv", index=False)
    print(f"\nWrote {out} and intensity_scan_summary.csv")


if __name__ == "__main__":
    main()
