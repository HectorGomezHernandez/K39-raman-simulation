"""
D1 line runs: linear 45 deg, pi+sigma, circular polarisations
across the active transitions, plus angle and elliptical scans.

Geometry: both beams in the xy-plane, symmetric about x_hat.
Beam 1 propagates at +HALF_ANGLE_DEG, beam 2 at -HALF_ANGLE_DEG.
Angle between propagation directions = BEAM_SEPARATION_DEG.

Beam separation can be passed on the command line:
    python run_D1.py 66.4
    python run_D1.py 133.6
    python run_D1.py 180

Outputs go to results/D1/{BEAM_SEPARATION_DEG}deg/.
"""
import os
import sys
import pandas as pd

from Raman_simulation import (
    D1,
    run_analysis,
    scan_polarization_angle,
    scan_polarization_ellipse,
    Beam,
)


if __name__ == "__main__":

    # Beam separation: angle between the two propagation directions.
    # Default = experimental setup (133.6 deg). Override via command line.
    BEAM_SEPARATION_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 133.6
    HALF_ANGLE_DEG = BEAM_SEPARATION_DEG / 2.0

    OUTPUT_DIR = os.path.join("results", "D1", f"{BEAM_SEPARATION_DEG}deg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)

    # Symmetric ±half-angle beam pairs for every polarization geometry.
    BEAM_LIN45_1 = Beam(theta_deg=+HALF_ANGLE_DEG, alpha_deg=45)
    BEAM_LIN45_2 = Beam(theta_deg=-HALF_ANGLE_DEG, alpha_deg=45)
    BEAM_PI_1    = Beam(theta_deg=+HALF_ANGLE_DEG, alpha_deg=0)   # pure pi
    BEAM_PERP_2  = Beam(theta_deg=-HALF_ANGLE_DEG, alpha_deg=90)  # linear perp B
    BEAM_CIRC_R_1 = Beam(theta_deg=+HALF_ANGLE_DEG, alpha_deg=45, phi_deg=90)
    BEAM_CIRC_R_2 = Beam(theta_deg=-HALF_ANGLE_DEG, alpha_deg=45, phi_deg=90)

    all_results = []
    # Active transitions. 1,1 -> 2,2 at 445.4 G dropped: 1.6 GHz ground splitting
    # exceeds the AOM detuning range in the experimental setup.
    transitions = [
        (402.7, 1.0, 0.0, "402.7G_mF1_to_mF0"),
        (445.4, 1.0, 0.0, "445.4G_mF1_to_mF0"),
        (526.0, -1.0, 0.0, "526G_mFm1_to_mF0"),
        (526.0, 0.0, 1.0, "526G_mF0_to_mF1"),
    ]

    # ================================================================
    # Part 1: Non-fixed waist, linear 45 deg (scan I and Delta)
    # ================================================================
    for B, mFi, mFf, label in transitions:
        all_results += run_analysis(
            B_G=B, mF_i=mFi, mF_f=mFf,
            label=f"{label}_freeWaist_lin45",
            beam1=BEAM_LIN45_1, beam2=BEAM_LIN45_2,
            line=D1,
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
            beam1=BEAM_LIN45_1, beam2=BEAM_LIN45_2,
            line=D1,
        )

    # ================================================================
    # Part 3: Polarisation exploration (fixed waist, 445.4 G, mF=1->0)
    # ================================================================

    # 3a: Linear angle scan
    scan_polarization_angle(
        B_G=445.4, mF_i=1.0, mF_f=0.0,
        Delta_MHz=840, w0_fixed_mm=1.0,
        theta1_deg=+HALF_ANGLE_DEG, theta2_deg=-HALF_ANGLE_DEG,
        label="445.4G_mF1_to_mF0_w0_1mm_lin",
        line=D1,
    )

    # 3b: Pi + sigma geometry, all three transitions
    for B, mFi, mFf, label in transitions:
        all_results += run_analysis(
            B_G=B, mF_i=mFi, mF_f=mFf,
            label=f"{label}_w0_1mm_pi_sigma",
            w0_fixed_mm=1.0,
            report_Delta_MHz=840,
            beam1=BEAM_PI_1, beam2=BEAM_PERP_2,
            line=D1,
        )

    # 3c: Circular polarisation (right circular, both beams)
    for B, mFi, mFf, label in transitions:
        all_results += run_analysis(
            B_G=B, mF_i=mFi, mF_f=mFf,
            label=f"{label}_w0_1mm_circular",
            w0_fixed_mm=1.0,
            report_Delta_MHz=840,
            beam1=BEAM_CIRC_R_1, beam2=BEAM_CIRC_R_2,
            line=D1,
        )

    # 3d: Elliptical polarisation 2D scan (alpha, phi)
    scan_polarization_ellipse(
        B_G=445.4, mF_i=1.0, mF_f=0.0,
        Delta_MHz=840, w0_fixed_mm=1.0,
        theta1_deg=+HALF_ANGLE_DEG, theta2_deg=-HALF_ANGLE_DEG,
        label="445.4G_mF1_to_mF0_w0_1mm",
        line=D1,
    )

    # 3e: Sanity check -- right + left circular gives destructive interference,
    # equivalent to linear 45 deg (both channels acquire opposite signs).
    # Verified at theta=0: t_pi = 30.93 us, P_sc = 0.394 (same as lin45). Uncomment to re-run.
    # BEAM_CIRC_L_2 = Beam(theta_deg=-HALF_ANGLE_DEG, alpha_deg=45, phi_deg=-90)
    # all_results += run_analysis(
    #     B_G=445.4, mF_i=1.0, mF_f=0.0,
    #     label="445.4G_mF1_to_mF0_w0_1mm_R_L",
    #     w0_fixed_mm=1.0,
    #     report_Delta_MHz=840,
    #     beam1=BEAM_CIRC_R_1, beam2=BEAM_CIRC_L_2,
    #     line=D1,
    # )

    # ================================================================
    # Comparison table
    # ================================================================
    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE (D1, beam separation = {BEAM_SEPARATION_DEG} deg)")
    print(f"{'='*80}")
    df_cmp = pd.DataFrame(all_results)
    cols = ["label", "B_G", "I_mWcm2", "Delta_MHz", "delta_req_2pi_MHz",
            "t_pi_us", "P_sc_avg", "omega_21_2pi_MHz",
            "w0_um", "omega_r_2pi_Hz", "k_kick_per_um"]
    table_str = df_cmp[cols].to_string(index=False)
    print(table_str)

    with open("comparison_table.txt", "w") as f:
        f.write(f"COMPARISON TABLE (D1, beam separation = {BEAM_SEPARATION_DEG} deg)\n")
        f.write(f"{'='*80}\n")
        f.write(table_str + "\n")
    print("\nSaved to comparison_table.txt")
