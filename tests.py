"""
Quick tests and sanity checks.
"""
import numpy as np
from scipy.constants import c, epsilon_0, hbar, pi, atomic_mass
from Raman_simulation import (
    D1, D2, evaluate_parameter_set, pol_linear, pol_elliptical_yz,
    pol_beam_xy, Beam, raman_recoil,
    diagonalize_manifold, dipole_operator_uncoupled,
    transform_operator_to_eigenbasis, reduced_dipole_from_linewidth,
    select_states_by_mF, excited_manifold_reference_energy,
    excited_state_offsets, C, m_K39,
)


def test_pol_beam_xy_reduces_to_yz(tol=1e-14):
    """
    Regression test: at theta = 0, pol_beam_xy(0, alpha, phi) must produce
    exactly the same spherical amplitudes as pol_elliptical_yz(alpha, phi).
    Also checks that the polarization is unit-norm for any (theta, alpha, phi).
    """
    cases = [(alpha, phi) for alpha in (0.0, 30.0, 45.0, 60.0, 90.0)
                          for phi   in (-90.0, -45.0, 0.0, 45.0, 90.0)]
    max_err_eq = 0.0
    for alpha, phi in cases:
        old = pol_elliptical_yz(alpha, phi)
        new = pol_beam_xy(0.0, alpha, phi)
        for q in (-1, 0, +1):
            err = abs(new[q] - old[q])
            max_err_eq = max(max_err_eq, err)

    # Norm check at several theta values: |c_pi|^2 + |c_s+|^2 + |c_s-|^2 == 1
    max_err_norm = 0.0
    for theta in (0.0, 30.0, 60.0, 90.0, 135.0, 180.0):
        for alpha, phi in cases:
            p = pol_beam_xy(theta, alpha, phi)
            norm2 = abs(p[0])**2 + abs(p[+1])**2 + abs(p[-1])**2
            max_err_norm = max(max_err_norm, abs(norm2 - 1.0))

    status_eq   = "OK" if max_err_eq   < tol else "FAIL"
    status_norm = "OK" if max_err_norm < tol else "FAIL"
    print(f"pol_beam_xy(theta=0) == pol_elliptical_yz [{status_eq}]: "
          f"max abs diff = {max_err_eq:.3e} (over {len(cases)} (alpha,phi) pairs)")
    print(f"pol_beam_xy norm == 1 [{status_norm}]: "
          f"max |norm^2 - 1| = {max_err_norm:.3e}")


def test_decay_sum_rule(line, B_G=1.0, tol=1e-10):
    """
    For every excited eigenstate, sum |<e|d_q|g>|^2 over all q and all ground
    eigenstates and apply the Einstein-A prefactor. Must recover Gamma_input,
    independent of B and of which eigenstate we pick.
    """
    _, _, U_g, basis_g = diagonalize_manifold(
        J=0.5, I=C.I_nuclear, ahfs=C.ahfs_S, gJ=C.gJ_S, gI=C.gI,
        B_G=B_G, manifold_label="S")
    _, _, U_e, basis_e = diagonalize_manifold(
        J=line.J_e, I=C.I_nuclear, ahfs=line.ahfs, gJ=line.gJ, gI=C.gI,
        B_G=B_G, manifold_label="P", bhfs=line.bhfs)

    d_red = reduced_dipole_from_linewidth(line.omega, line.gamma, Je=line.J_e)
    D_eig = {}
    for q in (-1, 0, +1):
        D_unc = dipole_operator_uncoupled(basis_e, basis_g, line.J_e, 0.5, q, d_red)
        D_eig[q] = transform_operator_to_eigenbasis(D_unc, U_e, U_g)

    prefactor = line.omega**3 / (3 * pi * epsilon_0 * hbar * c**3)
    n_e, n_g = D_eig[0].shape

    max_err = 0.0
    worst_ie = -1
    for ie in range(n_e):
        s = sum(abs(D_eig[q][ie, ig])**2 for q in (-1, 0, +1) for ig in range(n_g))
        Gamma_reco = prefactor * s
        rel_err = abs(Gamma_reco - line.gamma) / line.gamma
        if rel_err > max_err:
            max_err = rel_err
            worst_ie = ie

    status = "OK" if max_err < tol else "FAIL"
    print(f"{line.name} sum rule [{status}] at B={B_G} G: "
          f"worst rel err = {max_err:.3e} (state e{worst_ie}, "
          f"{n_e} excited states checked)")
    return max_err


def quick_eval(B_G, mF_i, mF_f, Delta_MHz, beam1, beam2, line=D1):
    """Evaluate a single (B, mF_i, mF_f, Delta) point and return the result dict."""
    gs, Eg, Ug, bg = diagonalize_manifold(
        J=0.5, I=C.I_nuclear, ahfs=C.ahfs_S, gJ=C.gJ_S, gI=C.gI,
        B_G=B_G, manifold_label="S")
    es, Ee, Ue, be = diagonalize_manifold(
        J=line.J_e, I=C.I_nuclear, ahfs=line.ahfs, gJ=line.gJ, gI=C.gI,
        B_G=B_G, manifold_label="P", bhfs=line.bhfs)
    d_red = reduced_dipole_from_linewidth(line.omega, line.gamma, Je=line.J_e)
    Dp = {}
    for q in (-1, 0, +1):
        Du = dipole_operator_uncoupled(be, bg, Je=line.J_e, Jg=0.5, q=q, reduced_matrix_element=d_red)
        Dp[q] = transform_operator_to_eigenbasis(Du, Ue, Ug)
    gi = select_states_by_mF(gs, mF_i)[0][0]
    gf = select_states_by_mF(gs, mF_f)[0][0]
    mF_int = sorted({mF_i + q for q in [-1, 0, 1]} & {mF_f + q for q in [-1, 0, 1]})
    ei = []
    for mf in mF_int:
        for ie, _ in select_states_by_mF(es, mf):
            ei.append(ie)
    Eref = excited_manifold_reference_energy(Ee)
    dn = excited_state_offsets(Ee, ei, Eref)
    return evaluate_parameter_set(
        159.155, Delta_MHz, 0.0, gi, gf, ei, Eg, Ee, dn, Dp, line.gamma,
        omega_laser=line.omega, beam1=beam1, beam2=beam2)


if __name__ == "__main__":

    # ================================================================
    # Test: decay sum rule (Wigner-Eckart consistency)
    # Pass means: dipole construction, reduced matrix element, basis
    # transformation are all internally consistent.
    # ================================================================
    print("=" * 70)
    print("Polarization helper: pol_beam_xy regression vs pol_elliptical_yz")
    print("=" * 70)
    test_pol_beam_xy_reduces_to_yz()

    print("\n" + "=" * 70)
    print("Decay sum rule test")
    print("=" * 70)
    for line in (D1, D2):
        for B_G in (1.0, 100.0, 402.7, 445.4):
            test_decay_sum_rule(line, B_G=B_G)

    # ================================================================
    # Test: Raman recoil. Copropagating must give |k_eff|=0; counter-
    # propagating along y (theta1=+90, theta2=-90) gives |k_eff|=2 omega/c.
    # For K-39 D1 that puts v_rec ~ 2.65 cm/s and omega_rec/2pi ~ 34 kHz.
    # ================================================================
    print("\n" + "=" * 70)
    print("Raman recoil test")
    print("=" * 70)
    for line in (D1, D2):
        b_co_1 = Beam(theta_deg=0.0)
        b_co_2 = Beam(theta_deg=0.0)
        rec_co = raman_recoil(b_co_1, b_co_2, line.omega, mass=m_K39)
        b_cp_1 = Beam(theta_deg=+90.0)
        b_cp_2 = Beam(theta_deg=-90.0)
        rec_cp = raman_recoil(b_cp_1, b_cp_2, line.omega, mass=m_K39)
        k_expected = 2 * line.omega / c * 1e-6  # 1/um
        v_expected_mm_s = hbar * (2 * line.omega / c) / m_K39 * 1e3
        omrec_expected_kHz = hbar * (2 * line.omega / c)**2 / (2 * m_K39) / (2 * pi) / 1e3
        print(f"\n{line.name}: omega_laser/2pi = {line.omega/(2*pi)/1e12:.3f} THz")
        print(f"  copropagating       : |k_eff| = {rec_co['k_eff_per_um']:.6e} 1/um (expected 0)")
        print(f"  counter-propagating : |k_eff| = {rec_cp['k_eff_per_um']:.6f} 1/um "
              f"(expected {k_expected:.6f})")
        print(f"                        v_rec   = {rec_cp['v_rec_mm_s']:.4f} mm/s "
              f"(expected {v_expected_mm_s:.4f})")
        print(f"                        omega_rec/2pi = {rec_cp['omega_rec_2pi_kHz']:.4f} kHz "
              f"(expected {omrec_expected_kHz:.4f})")

    # ================================================================
    # Test: sigma + pi vs pi + sigma vs circular
    # Expect: pi+sigma > circular > sigma+pi at Delta = 840 MHz
    # ================================================================

    BEAM_PI = Beam(alpha_deg=0)         # pure pi
    BEAM_PERP = Beam(alpha_deg=90)      # linear perpendicular to B
    BEAM_CIRC_R = Beam(alpha_deg=45, phi_deg=90)  # right circular

    cases = [
        (402.7, 1.0, 0.0, "402.7 G, mF=1->0"),
        (445.4, 1.0, 0.0, "445.4 G, mF=1->0"),
        (445.4, 1.0, 2.0, "445.4 G, mF=1->2"),
    ]

    geometries = [
        ("pi + sigma", BEAM_PI, BEAM_PERP),
        ("circular (R+R)", BEAM_CIRC_R, BEAM_CIRC_R),
        ("sigma + pi", BEAM_PERP, BEAM_PI),
    ]

    for B, mFi, mFf, case_label in cases:
        print(f"\nD1, {case_label}, Delta=840 MHz, w0=1mm (I=159.2 mW/cm^2)")
        print("=" * 70)
        print(f"{'Geometry':25s} {'t_pi (us)':>12s} {'P_sc_avg':>12s} {'|Omega_R|/2pi':>15s}")
        print("-" * 70)

        for name, b1, b2 in geometries:
            res = quick_eval(B, mFi, mFf, 840, b1, b2, line=D1)
            print(f"{name:25s} {res['t_pi_us']:12.3f} {res['P_sc_avg']:12.4f} {res['Omega_R_abs_2pi_MHz']:15.6f} MHz")
