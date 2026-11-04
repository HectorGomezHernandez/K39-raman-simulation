"""
Overlay t_pi and P_sc curves under old (filtered-intermediates) vs new
(full-manifold) reference for the excited-manifold energy.

If the change is purely a redefinition of the zero of Delta, then
old(Delta) == new(Delta - shift), where shift = E_ref_old - E_ref_new.
Visualised as: the green dotted curve (new shifted) must coincide with
the blue curve (old).

Outputs one PNG per (line, B, mF transition, geometry) into the same
folder as this script.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT)

from Raman_simulation import (
    D1, D2, C, diagonalize_manifold, dipole_operator_uncoupled,
    transform_operator_to_eigenbasis, reduced_dipole_from_linewidth,
    select_states_by_mF, excited_manifold_reference_energy,
    excited_state_offsets, evaluate_parameter_set, Beam,
)


def precompute(line, B_G, mF_i, mF_f):
    gs, Eg, Ug, bg = diagonalize_manifold(
        J=0.5, I=C.I_nuclear, ahfs=C.ahfs_S, gJ=C.gJ_S, gI=C.gI,
        B_G=B_G, manifold_label="S")
    es, Ee, Ue, be = diagonalize_manifold(
        J=line.J_e, I=C.I_nuclear, ahfs=line.ahfs, gJ=line.gJ, gI=C.gI,
        B_G=B_G, manifold_label="P", bhfs=line.bhfs)
    d_red = reduced_dipole_from_linewidth(line.omega, line.gamma, Je=line.J_e)
    Dp = {}
    for q in (-1, 0, +1):
        Du = dipole_operator_uncoupled(be, bg, line.J_e, 0.5, q, d_red)
        Dp[q] = transform_operator_to_eigenbasis(Du, Ue, Ug)
    gi = select_states_by_mF(gs, mF_i)[0][0]
    gf = select_states_by_mF(gs, mF_f)[0][0]
    mF_int = sorted({mF_i + q for q in (-1, 0, 1)} & {mF_f + q for q in (-1, 0, 1)})
    ei = [i for mf in mF_int for i, _ in select_states_by_mF(es, mf)]
    return Eg, Ee, Dp, gi, gf, ei, line.gamma


def scan(line, B_G, mF_i, mF_f, beam1, beam2, Delta_arr, use_full_manifold):
    Eg, Ee, Dp, gi, gf, ei, Gamma = precompute(line, B_G, mF_i, mF_f)
    if use_full_manifold:
        E_ref = excited_manifold_reference_energy(Ee)
    else:
        E_ref = excited_manifold_reference_energy(Ee, ei)
    dn = excited_state_offsets(Ee, ei, E_ref)
    tpi = np.zeros_like(Delta_arr)
    psc = np.zeros_like(Delta_arr)
    I = 159.154943  # mW/cm^2 per beam, P_total=5 mW, w0=1 mm
    for k, D_MHz in enumerate(Delta_arr):
        r = evaluate_parameter_set(I, D_MHz, 0.0, gi, gf, ei,
                                   Eg, Ee, dn, Dp, Gamma,
                                   omega_laser=line.omega,
                                   beam1=beam1, beam2=beam2)
        tpi[k] = r["t_pi_us"]
        psc[k] = r["P_sc_avg"]
    shift_MHz = (excited_manifold_reference_energy(Ee, ei)
                 - excited_manifold_reference_energy(Ee)) / (2 * np.pi * 1e6)
    return tpi, psc, shift_MHz


BEAM_PI = Beam(alpha_deg=0)
BEAM_PERP = Beam(alpha_deg=90)
Delta_arr = np.linspace(-3000, 3000, 800)

cases = [
    (D1, 402.7, 1.0, 0.0, "D1_402.7G_mF1_to_mF0_pi_sigma"),
    (D1, 445.4, 1.0, 2.0, "D1_445.4G_mF1_to_mF2_pi_sigma"),
    (D2, 402.7, 1.0, 0.0, "D2_402.7G_mF1_to_mF0_pi_sigma"),
    (D2, 445.4, 1.0, 2.0, "D2_445.4G_mF1_to_mF2_pi_sigma"),
]

for line, B, mFi, mFf, fname in cases:
    tpi_old, psc_old, shift = scan(line, B, mFi, mFf, BEAM_PI, BEAM_PERP,
                                    Delta_arr, use_full_manifold=False)
    tpi_new, psc_new, _ = scan(line, B, mFi, mFf, BEAM_PI, BEAM_PERP,
                                Delta_arr, use_full_manifold=True)

    fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax_t.plot(Delta_arr, np.clip(tpi_old, 0, 100), color="C0",
              label="old reference (filtered intermediates)", linewidth=1.8)
    ax_t.plot(Delta_arr, np.clip(tpi_new, 0, 100), color="C1",
              label="new reference (full manifold)", linewidth=1.8, linestyle="--")
    ax_t.plot(Delta_arr - shift, np.clip(tpi_new, 0, 100), color="C2",
              label=f"new shifted by {shift:+.2f} MHz",
              linewidth=1.2, linestyle=":")
    ax_t.set_ylabel("t_pi (us)")
    ax_t.set_ylim(0, 100)
    ax_t.set_title(f"{fname}   (shift old - new = {shift:+.2f} MHz)")
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="upper right")

    ax_p.plot(Delta_arr, np.clip(psc_old, 0, 1), color="C0",
              label="old", linewidth=1.8)
    ax_p.plot(Delta_arr, np.clip(psc_new, 0, 1), color="C1",
              label="new", linewidth=1.8, linestyle="--")
    ax_p.plot(Delta_arr - shift, np.clip(psc_new, 0, 1), color="C2",
              label="new shifted", linewidth=1.2, linestyle=":")
    ax_p.set_ylabel("P_sc_avg")
    ax_p.set_xlabel("Delta (MHz)")
    ax_p.set_ylim(0, 1)
    ax_p.axhline(0.05, color="green", linestyle="--", alpha=0.4)
    ax_p.axhline(0.10, color="orange", linestyle="--", alpha=0.4)
    ax_p.grid(alpha=0.3)
    ax_p.legend(loc="upper right")

    plt.tight_layout()
    out = os.path.join(HERE, f"old_vs_new_{fname}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}  (shift = {shift:+.2f} MHz)")
