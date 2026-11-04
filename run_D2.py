"""
D2 line runs: same configurations as run_D1.py but using the D2 line (P3/2).
Outputs go to results/D2/.
"""
import os
import pandas as pd

from Raman_simulation import (
    D2,
    run_analysis,
    scan_polarization_angle,
    scan_polarization_ellipse,
    Beam,
)


if __name__ == "__main__":

    OUTPUT_DIR = os.path.join("results", "D2")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)

    all_results = []
    transitions = [
        (402.7, 1.0, 0.0, "402.7G_mF1_to_mF0"),
        (445.4, 1.0, 0.0, "445.4G_mF1_to_mF0"),
        (445.4, 1.0, 2.0, "445.4G_mF1_to_mF2"),
    ]

    # ================================================================
    # Part 1: Non-fixed waist, linear 45 deg (scan I and Delta)
    # ================================================================
    for B, mFi, mFf, label in transitions:
        all_results += run_analysis(
            B_G=B, mF_i=mFi, mF_f=mFf,
            label=f"{label}_freeWaist_lin45",
            line=D2,
        )

    # ================================================================
    # Part 2: Fixed waist w0=1mm, linear 45 deg, report at 840 MHz
    # ================================================================
    for B, mFi, mFf, label in transitions:
        all_results += run_analysis(
            B_G=B, mF_i=mFi, mF_f=mFf,
            label=f"{label}_w0_1mm_lin45",
            w0_fixed_mm=1.0,
            report_Delta_MHz=840,
            line=D2,
        )

    # ================================================================
    # Part 3: Polarisation exploration (fixed waist, 402.7 G, mF=1->0)
    # ================================================================

    # 3a: Linear angle scan
    scan_polarization_angle(
        B_G=402.7, mF_i=1.0, mF_f=0.0,
        Delta_MHz=840, w0_fixed_mm=1.0,
        label="402.7G_mF1_to_mF0_w0_1mm_lin",
        line=D2,
    )

    # 3b: Pi + sigma geometry, all three transitions
    BEAM_PI = Beam(alpha_deg=0)
    BEAM_PERP = Beam(alpha_deg=90)
    for B, mFi, mFf, label in transitions:
        all_results += run_analysis(
            B_G=B, mF_i=mFi, mF_f=mFf,
            label=f"{label}_w0_1mm_pi_sigma",
            w0_fixed_mm=1.0,
            report_Delta_MHz=840,
            beam1=BEAM_PI, beam2=BEAM_PERP,
            line=D2,
        )

    # 3c: Circular polarisation (right circular, both beams)
    BEAM_CIRC_R = Beam(alpha_deg=45, phi_deg=90)
    for B, mFi, mFf, label in transitions:
        all_results += run_analysis(
            B_G=B, mF_i=mFi, mF_f=mFf,
            label=f"{label}_w0_1mm_circular",
            w0_fixed_mm=1.0,
            report_Delta_MHz=840,
            beam1=BEAM_CIRC_R, beam2=BEAM_CIRC_R,
            line=D2,
        )

    # 3d: Elliptical polarisation 2D scan (alpha, phi)
    scan_polarization_ellipse(
        B_G=402.7, mF_i=1.0, mF_f=0.0,
        Delta_MHz=840, w0_fixed_mm=1.0,
        label="402.7G_mF1_to_mF0_w0_1mm",
        line=D2,
    )

    # 3e: Sanity check -- right + left circular gives destructive interference,
    # equivalent to linear 45 deg (both channels acquire opposite signs).
    # Verified for D1 (t_pi = 30.93 us, P_sc = 0.394, same as lin45). Uncomment to re-run.
    # BEAM_CIRC_L = Beam(alpha_deg=45, phi_deg=-90)
    # all_results += run_analysis(
    #     B_G=402.7, mF_i=1.0, mF_f=0.0,
    #     label="402.7G_mF1_to_mF0_w0_1mm_R_L",
    #     w0_fixed_mm=1.0,
    #     report_Delta_MHz=840,
    #     beam1=BEAM_CIRC_R, beam2=BEAM_CIRC_L,
    #     line=D2,
    # )

    # ================================================================
    # Comparison table
    # ================================================================
    print(f"\n{'='*80}")
    print("COMPARISON TABLE (D2)")
    print(f"{'='*80}")
    df_cmp = pd.DataFrame(all_results)
    cols = ["label", "B_G", "I_mWcm2", "Delta_MHz", "delta_req_2pi_MHz",
            "t_pi_us", "P_sc_avg", "omega_21_2pi_MHz",
            "w0_um", "omega_r_2pi_Hz", "k_kick_per_um"]
    table_str = df_cmp[cols].to_string(index=False)
    print(table_str)

    with open("comparison_table.txt", "w") as f:
        f.write("COMPARISON TABLE (D2)\n")
        f.write(f"{'='*80}\n")
        f.write(table_str + "\n")
    print("\nSaved to comparison_table.txt")
