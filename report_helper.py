"""
Compute, for each (beam separation, transition, polarization) combination
at fixed waist 1 mm:

  - Delta where P_sc(Delta) = 5% and 10% (linear interpolation from a fine
    fixed-waist scan across positive Delta).
  - t_pi, P_sc, delta_AC, omega_rec, delta_req at Delta = 1.2 GHz.

Output: report_summary.txt next to the script, grouped by separation.
"""
import numpy as np
import pandas as pd

from Raman_simulation import (
    D1, Beam, diagonalize_manifold, dipole_operator_uncoupled,
    transform_operator_to_eigenbasis, reduced_dipole_from_linewidth,
    select_states_by_mF, excited_manifold_reference_energy,
    excited_state_offsets, evaluate_parameter_set, C,
)


I_FIXED_MWCM2 = 159.154943   # P_total = 5 mW, w0 = 1 mm
DELTA_REPORT_MHZ_LIST = [700.0, 1200.0]
DELTA_GRID = np.linspace(50.0, 3000.0, 1000)


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
    return Eg, Ee, Dp, gi, gf, ei


def first_crossing(x, y, target):
    """First Delta where P_sc(Delta) crosses target, or None."""
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
    (402.7, 1.0, 0.0, "402.7 G, mF: 1->0"),
    (445.4, 1.0, 0.0, "445.4 G, mF: 1->0"),
    (526.0, -1.0, 0.0, "526 G, mF: -1->0"),
    (526.0, 0.0, 1.0, "526 G, mF: 0->1"),
]

polarizations = [
    ("lin45",     45, 0,  45, 0),
    ("pi+sigma",   0, 0,  90, 0),
    ("circular",  45, 90, 45, 90),
]

separations = [0.0, 66.4, 133.6, 180.0]


def evaluate_at(Delta_MHz, line, Eg, Ee, Dp, gi, gf, ei, beam1, beam2):
    return evaluate_parameter_set(
        I_FIXED_MWCM2, Delta_MHz, 0.0, gi, gf, ei,
        Eg, Ee, excited_state_offsets(Ee, ei, excited_manifold_reference_energy(Ee)),
        Dp, line.gamma, omega_laser=line.omega, beam1=beam1, beam2=beam2,
    )


def main():
    rows = []
    for sep in separations:
        for B_G, mFi, mFf, tlabel in transitions:
            Eg, Ee, Dp, gi, gf, ei = precompute(D1, B_G, mFi, mFf)
            for plabel, a1, p1, a2, p2 in polarizations:
                beam1, beam2 = make_beam_pair(sep, a1, p1, a2, p2)
                # Delta scan to find P_sc = 5% / 10% crossings
                psc = np.array([
                    evaluate_at(D, D1, Eg, Ee, Dp, gi, gf, ei, beam1, beam2)["P_sc_avg"]
                    for D in DELTA_GRID
                ])
                D_5 = first_crossing(DELTA_GRID, psc, 0.05)
                D_10 = first_crossing(DELTA_GRID, psc, 0.10)
                row = {
                    "sep_deg": sep,
                    "transition": tlabel,
                    "pol": plabel,
                    "Delta_5pct_MHz": D_5,
                    "Delta_10pct_MHz": D_10,
                }
                # Values at each report Delta
                report_strs = []
                for D_MHz in DELTA_REPORT_MHZ_LIST:
                    r = evaluate_at(D_MHz, D1, Eg, Ee, Dp, gi, gf, ei, beam1, beam2)
                    tag = f"{D_MHz/1000:g}GHz" if D_MHz >= 1000 else f"{D_MHz:g}MHz"
                    row[f"t_pi_us@{tag}"]       = r["t_pi_us"]
                    row[f"P_sc@{tag}"]          = r["P_sc_avg"]
                    row[f"delta_AC_MHz@{tag}"]  = r["delta_AC_2pi_MHz"]
                    row[f"delta_req_MHz@{tag}"] = r["delta_req_2pi_MHz"]
                    report_strs.append(
                        f"{tag}: t={r['t_pi_us']:5.2f}us P={r['P_sc_avg']:.4f}")
                # Recoil quantities (geometry-dependent only, not Delta)
                row["omega_rec_kHz"] = r["omega_rec_2pi_kHz"]
                row["k_eff_per_um"]  = r["k_eff_per_um"]
                row["v_rec_mm_s"]    = r["v_rec_mm_s"]
                rows.append(row)
                print(f"{sep:6.1f} | {tlabel:18s} | {plabel:8s} | "
                      f"D5={('---' if D_5 is None else f'{D_5:7.1f}')} | "
                      f"D10={('---' if D_10 is None else f'{D_10:7.1f}')} | "
                      + "  ".join(report_strs))

    df = pd.DataFrame(rows)
    out = "report_summary.txt"
    with open(out, "w") as f:
        f.write(f"Fixed waist w0 = 1 mm, I per beam = {I_FIXED_MWCM2:.3f} mW/cm^2\n")
        f.write(f"Delta_5pct_MHz, Delta_10pct_MHz: smallest positive Delta in [50, 3000] MHz\n")
        f.write(f"  where P_sc(Delta) crosses the threshold (linearly interpolated,\n")
        f.write(f"  None if never reached).\n")
        f.write(f"Per-Delta columns are at Delta in {DELTA_REPORT_MHZ_LIST} MHz.\n")
        f.write("=" * 100 + "\n")
        f.write(df.to_string(index=False) + "\n")
    df.to_csv("report_summary.csv", index=False)
    print(f"\nWrote {out} and report_summary.csv")


if __name__ == "__main__":
    main()
