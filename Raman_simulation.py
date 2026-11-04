import numpy as np
from scipy.constants import c, epsilon_0, hbar, pi, atomic_mass
from dataclasses import dataclass
from scipy.linalg import eigh
from sympy import S
from sympy.physics.wigner import wigner_3j
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# ============================================================
# Constants
# ============================================================

def _eval_const(val):
    if isinstance(val, str):
        return eval(val, {"pi": np.pi})
    return val

with open(os.path.join(os.path.dirname(__file__), "constants.json")) as f:
    _raw = json.load(f)

mu_B_over_hbar = _eval_const(_raw["mu_B_over_hbar"])

@dataclass
class Constants:
    I_nuclear: float = _eval_const(_raw["I_nuclear"])
    gI: float = _eval_const(_raw["gI"])
    gJ_S: float = _eval_const(_raw["gJ_S"])
    gJ_P1_2: float = _eval_const(_raw["gJ_P1_2"])
    gJ_P3_2: float = _eval_const(_raw["gJ_P3_2"])
    ahfs_S: float = _eval_const(_raw["ahfs_S"])
    ahfs_D1: float = _eval_const(_raw["ahfs_D1"])
    ahfs_D2: float = _eval_const(_raw["ahfs_D2"])
    bhfs_D2: float = _eval_const(_raw["bhfs_D2"])
    gamma_D1: float = _eval_const(_raw["gamma_D1"])
    gamma_D2: float = _eval_const(_raw["gamma_D2"])
    omega_D1: float = _eval_const(_raw["omega_D1"])
    omega_D2: float = _eval_const(_raw["omega_D2"])

C = Constants()


@dataclass
class LineConfig:
    """Bundles the parameters specific to one optical transition (D1 or D2)."""
    name: str
    J_e: float
    ahfs: float
    bhfs: float  # quadrupole hyperfine constant (0 if J<1 or I<1)
    gJ: float
    omega: float
    gamma: float


D1 = LineConfig(
    name="D1",
    J_e=0.5,
    ahfs=C.ahfs_D1,
    bhfs=0.0,           # P1/2 has no quadrupole (J=1/2)
    gJ=C.gJ_P1_2,
    omega=C.omega_D1,
    gamma=C.gamma_D1,
)

D2 = LineConfig(
    name="D2",
    J_e=1.5,
    ahfs=C.ahfs_D2,
    bhfs=C.bhfs_D2,
    gJ=C.gJ_P3_2,
    omega=C.omega_D2,
    gamma=C.gamma_D2,
)


# ============================================================
# Basis states
# ============================================================

@dataclass(frozen=True)
class UncoupledBasisState:
    mJ: float
    mI: float

    @property
    def mF(self):
        return self.mJ + self.mI


@dataclass
class DiagonalizedState:
    manifold: str
    energy: float
    mF_expectation: float
    coeffs: np.ndarray
    basis: list


def basis_mJ_mI(J: float, I: float):
    """
    Uncoupled basis |mJ, mI>.
    Order:
    mJ = -J, -J+1, ..., J
    for each mJ, mI = -I, -I+1, ..., I
    """
    mJs = np.arange(-J, J + 1, 1.0)
    mIs = np.arange(-I, I + 1, 1.0)
    return [UncoupledBasisState(float(mJ), float(mI)) for mJ in mJs for mI in mIs]

# ============================================================
# Angular momentum matrices
# ============================================================

def j_matrices(j: float):
    """
    Return Jx, Jy, Jz in the standard |mJ> basis with mJ ascending:
    m = -J, -J+1, ..., J
    Will return also Ix, Iy, Iz in the standart |mI> basis with mI ascending.
    """
    m_vals = np.arange(-j, j + 1, 1.0)
    dim = len(m_vals)

    Jp = np.zeros((dim, dim), dtype=complex)
    Jm = np.zeros((dim, dim), dtype=complex)
    Jz = np.diag(m_vals)

    index_of_m = {m: i for i, m in enumerate(m_vals)}

    for m in m_vals:
        i = index_of_m[m]

        mp = m + 1
        if mp <= j:
            ip = index_of_m[mp]
            Jp[ip, i] = np.sqrt(j * (j + 1) - m * (m + 1))

        mm = m - 1
        if mm >= -j:
            im = index_of_m[mm]
            Jm[im, i] = np.sqrt(j * (j + 1) - m * (m - 1))

    Jx = 0.5 * (Jp + Jm)
    Jy = (Jp - Jm) / (2j)
    return Jx, Jy, Jz

# ============================================================
# Hyperfine + Zeeman Hamiltonian diagonalization
# ============================================================

def manifold_hamiltonian(J, I, ahfs, gJ, gI, B_G, bhfs=0.0):
    """
    Hamiltonian in the uncoupled basis |mJ,mI>:
        H = ahfs I·J
          + bhfs [3(I·J)^2 + (3/2)(I·J) - I^2 J^2] / [2I(2I-1)J(2J-1)]
          + mu_B/hbar * B * (gJ Jz + gI Iz)

    The quadrupole term (bhfs) is included only if bhfs != 0.
    """
    Jx, Jy, Jz = j_matrices(J)
    Ix, Iy, Iz = j_matrices(I)

    dimJ = Jx.shape[0]
    dimI = Ix.shape[0]

    Jx_full = np.kron(Jx, np.eye(dimI))
    Jy_full = np.kron(Jy, np.eye(dimI))
    Jz_full = np.kron(Jz, np.eye(dimI))

    Ix_full = np.kron(np.eye(dimJ), Ix)
    Iy_full = np.kron(np.eye(dimJ), Iy)
    Iz_full = np.kron(np.eye(dimJ), Iz)

    I_dot_J = Jx_full @ Ix_full + Jy_full @ Iy_full + Jz_full @ Iz_full
    H_hfs_dipole = ahfs * I_dot_J

    if bhfs != 0.0:
        J_squared = Jx_full @ Jx_full + Jy_full @ Jy_full + Jz_full @ Jz_full
        I_squared = Ix_full @ Ix_full + Iy_full @ Iy_full + Iz_full @ Iz_full
        quad_num = 3 * (I_dot_J @ I_dot_J) + 1.5 * I_dot_J - I_squared @ J_squared
        quad_den = 2 * I * (2 * I - 1) * J * (2 * J - 1)
        H_hfs_quad = bhfs * quad_num / quad_den
    else:
        H_hfs_quad = 0

    H_Z = mu_B_over_hbar * B_G * (gJ * Jz_full + gI * Iz_full)

    return H_hfs_dipole + H_hfs_quad + H_Z

def diagonalize_manifold(J: float, I: float, ahfs: float, gJ: float, gI: float, B_G: float, manifold_label: str, bhfs: float = 0.0):
    """
    Diagonalize one manifold Hamiltonian.
    Returns:
    states: list of DiagonalizedState
    energies: ndarray
    U: eigenvector matrix, columns are eigenvectors in uncoupled basis
    basis: uncoupled basis
    """
    basis = basis_mJ_mI(J, I)
    H = manifold_hamiltonian(J, I, ahfs, gJ, gI, B_G, bhfs=bhfs)
    energies, U = eigh(H)

    states = []
    for k in range(len(energies)):
        coeffs = U[:, k]

        mF_exp = 0.0
        for idx, st in enumerate(basis):
            mF_exp += (abs(coeffs[idx]) ** 2) * st.mF

        states.append(
            DiagonalizedState(
            manifold=manifold_label,
            energy=float(energies[k]),
            mF_expectation=float(mF_exp),
            coeffs=coeffs.copy(),
            basis=basis,
            )
        )

    return states, energies, U, basis

# ============================================================
# Dipole matrix elements in uncoupled basis
# ============================================================

# Transform float values into exact angular momentum values
def _to_sympy_halfint(x: float):
    return S(int(round(2 * x))) / 2


def electronic_dipole_element(
    Je: float,
    me: float,
    Jg: float,
    mg: float,
    q: int,
    reduced_matrix_element: float = 1.0,
) -> complex:
    """
    <Je,me | d_q | Jg,mg>
    Wigner-Eckart convention:
    <Je me | T^(1)_q | Jg mg>
    = (-1)^(Je-me) * (Je 1 Jg; -me q mg) * <Je||T||Jg>
    """
    if abs(me - mg - q) > 1e-12:
        return 0.0 + 0.0j

    Je_s = _to_sympy_halfint(Je)
    me_s = _to_sympy_halfint(me)
    Jg_s = _to_sympy_halfint(Jg)
    mg_s = _to_sympy_halfint(mg)
    q_s = S(q)

    three_j = wigner_3j(Je_s, S(1), Jg_s, -me_s, q_s, mg_s)
    if three_j == 0:
        return 0.0 + 0.0j

    phase = (-1) ** int(round(Je - me))
    return phase * complex(three_j.evalf()) * reduced_matrix_element


def uncoupled_dipole_element(
    excited_state: UncoupledBasisState,
    ground_state: UncoupledBasisState,
    Je: float,
    Jg: float,
    q: int,
    reduced_matrix_element: float = 1.0,
) -> complex:
    """
    <Je,mJ',mI' | d_q | Jg,mJ,mI>
    Dipole acts only on electronic coordinates, so mI is conserved.
    """
    if abs(excited_state.mI - ground_state.mI) > 1e-12:
        return 0.0 + 0.0j

    return electronic_dipole_element(
        Je=Je,
        me=excited_state.mJ,
        Jg=Jg,
        mg=ground_state.mJ,
        q=q,
        reduced_matrix_element=reduced_matrix_element,
    )


def dipole_operator_uncoupled(
    basis_excited,
    basis_ground,
    Je: float,
    Jg: float,
    q: int,
    reduced_matrix_element: float = 1.0,
):
    """
    Matrix D_q with entries
    D_q[e,g] = <e_uncoupled | d_q | g_uncoupled>
    """
    D = np.zeros((len(basis_excited), len(basis_ground)), dtype=complex)

    for ie, e in enumerate(basis_excited):
        for ig, g in enumerate(basis_ground):
            D[ie, ig] = uncoupled_dipole_element(
                excited_state=e,
                ground_state=g,
                Je=Je,
                Jg=Jg,
                q=q,
                reduced_matrix_element=reduced_matrix_element,
            )
    return D


def transform_operator_to_eigenbasis(
    D_uncoupled: np.ndarray,
    U_excited: np.ndarray,
    U_ground: np.ndarray,
):
    """
    D_eig[e,g] = <e_eigen | d_q | g_eigen>
    with columns of U_* = eigenvectors in uncoupled basis.
    """
    return U_excited.conj().T @ D_uncoupled @ U_ground


# ============================================================
# Helpers to inspect states and couplings
# ============================================================

def dominant_components(state: DiagonalizedState, nmax: int = 2):
    """
    Return the nmax largest uncoupled basis components of a diagonalized state.
    """
    weights = np.abs(state.coeffs) ** 2
    idxs = np.argsort(weights)[::-1][:nmax]
    out = []
    for i in idxs:
        st = state.basis[i]
        out.append((i, st.mJ, st.mI, st.mF, weights[i]))
    return out


def print_states(states, title="", nmax=2):
    if title:
        print(f"\n=== {title} ===")
    for i, st in enumerate(states):
        print(f"\nState {i}: E/2π = {st.energy / (2*np.pi) / 1e6: .6f} MHz, <mF> = {st.mF_expectation: .6f}")
        for idx, mJ, mI, mF, w in dominant_components(st, nmax=nmax):
            print(f" basis idx {idx:2d}: |mJ={mJ:+.1f}, mI={mI:+.1f}> (mF={mF:+.1f}) weight={w:.6f}")


def save_states_to_json(states, filename, title="", nmax=2):
    states_data = []
    for i, st in enumerate(states):
        comps = []
        for idx, mJ, mI, mF, w in dominant_components(st, nmax=nmax):
            comps.append({
                "basis_idx": int(idx),
                "mJ": mJ,
                "mI": mI,
                "mF": mF,
                "weight": w,
            })
        states_data.append({
            "state_index": i,
            "energy_over_2pi_MHz": st.energy / (2 * np.pi) / 1e6,
            "mF_expectation": st.mF_expectation,
            "dominant_components": comps,
        })
    with open(filename, "w") as f:
        json.dump({"title": title, "states": states_data}, f, indent=4)


def select_states_by_mF(states, target_mF, tol=1e-6):
    return [(i, st) for i, st in enumerate(states) if abs(st.mF_expectation - target_mF) < tol]


def print_nonzero_uncoupled_elements(D, basis_e, basis_g, label, thresh=1e-12):
    print(f"\n=== Nonzero uncoupled elements for {label} ===")
    for ie, e in enumerate(basis_e):
        for ig, g in enumerate(basis_g):
            val = D[ie, ig]
            if abs(val) > thresh:
                print(
                    f"<e idx {ie:2d} |mJ={e.mJ:+.1f},mI={e.mI:+.1f}| "
                    f"{label} |g idx {ig:2d} |mJ={g.mJ:+.1f},mI={g.mI:+.1f}>"
                    f" = {val}"
                )


def print_selected_eigen_couplings(D_eig, excited_states, ground_states, excited_indices, ground_indices, label):
    print(f"\n=== Eigenbasis couplings for {label} ===")
    for ie in excited_indices:
        est = excited_states[ie]
        for ig in ground_indices:
            gst = ground_states[ig]
            val = D_eig[ie, ig]
            print(
                f"<e{ie}|{label}|g{ig}> = {val.real:+.6e}{val.imag:+.6e}j "
                f"(e mF≈{est.mF_expectation:+.1f}, g mF≈{gst.mF_expectation:+.1f})"
            )

# ============================================================
# Physical unit conversions
# ============================================================

def reduced_dipole_from_linewidth(omega0, Gamma, Je):
    """
    Compute the reduced matrix element from spontaneous decay rate.

    Parameters
    ----------
    omega0 : float
    Transition angular frequency [rad/s]
    Gamma : float
    Spontaneous decay rate [1/s]
    Je : float
    Upper-state electronic angular momentum

    Returns
    -------
    float
    Reduced dipole matrix element [C*m]
    """
    return np.sqrt(Gamma * 3 * pi * epsilon_0 * hbar * c**3 * (2 * Je + 1) / omega0**3)

def electric_field_amplitude_from_intensity(I_mWcm2):
    """
    Electric-field amplitude E0 from intensity I:
    I = (1/2) c eps0 E0^2

    Parameters
    ----------
    I_mWcm2 : float
    Intensity in mW/cm^2

    Returns
    -------
    float
    E0 in V/m
    """
    I_Wm2 = I_mWcm2*10
    return np.sqrt(2 * I_Wm2 / (c * epsilon_0))


# NOTE: dead code, kept for standalone use (e.g. quick checks in a notebook).
# The physics functions now compute Rabi frequencies directly from complex E-field
# amplitudes (Om = -E_q * d / hbar), so this helper is no longer called internally.
def rabi_from_matrix_element(d_eg, I_mWcm2):
    """
    One-photon Rabi frequency:
    Omega = E0 * d_eg / hbar

    Parameters
    ----------
    d_eg : complex
    Dipole matrix element [C*m]
    I_mWcm2 : float
    Intensity in mW/cm^2

    Returns
    -------
    complex
    Rabi frequency [rad/s]
    """
    E0 = electric_field_amplitude_from_intensity(I_mWcm2)
    return - E0 * d_eg / hbar

# ============================================================
# Gaussian beam focusing and dipole kick
# ============================================================

m_K39 = 39 * atomic_mass  # K-39 mass [kg]

def beam_waist_from_power(P_total_mW, I_per_beam_mWcm2):
    """
    Beam waist w0 for a Gaussian beam, assuming 50/50 power split.
    I_peak = 2P / (pi w0^2)  =>  w0 = sqrt(2P / (pi I_peak))
    """
    P_per_beam_W = (P_total_mW / 2) * 1e-3
    I_Wm2 = I_per_beam_mWcm2 * 10
    return np.sqrt(2 * P_per_beam_W / (np.pi * I_Wm2))


def dipole_kick(omega_AC_rad_s, w0_m, mass_kg, L_m, t_pi_s):
    """
    Momentum kick from the parasitic dipole potential of a focused Gaussian beam.

    Radial trapping frequency: omega_r = sqrt(4 |U0| / (m w0^2))
    with U0 = hbar * omega_AC.

    Kick at cloud edge: k_kick = m omega_r^2 L t_pi / hbar
    """
    U0 = hbar * abs(omega_AC_rad_s)
    omega_r = np.sqrt(4 * U0 / (mass_kg * w0_m**2))
    k_kick = mass_kg * omega_r**2 * L_m * t_pi_s / hbar
    return {
        "w0_um": w0_m * 1e6,
        "omega_r_2pi_Hz": omega_r / (2 * np.pi),
        "k_kick_per_um": k_kick * 1e-6,
    }


def print_kick_report(kick):
    print(f"  w0 = {kick['w0_um']:.2f} um")
    print(f"  omega_r /2π = {kick['omega_r_2pi_Hz']:.2f} Hz")
    print(f"  k_kick = {kick['k_kick_per_um']:.4e} um^-1")


# ============================================================
# Multilevel Raman frequency, AC Stark Shift
# ============================================================

def excited_manifold_reference_energy(E_e, excited_indices=None):
    """
    Reference energy for the excited manifold = mean of all eigenstate energies
    in the diagonalized manifold. Independent of which intermediates are selected
    for a given Raman transition, so Delta has the same physical meaning across
    transitions on a given line.

    If excited_indices is provided, average only over that subset (legacy
    behaviour, kept so existing callers don't break).
    """
    if excited_indices is None:
        vals = np.asarray(E_e, dtype=float)
    else:
        vals = np.array([E_e[i] for i in excited_indices], dtype=float)
    return float(np.mean(vals))


def excited_state_offsets(E_e, excited_indices, E_ref_exc):
    """
    Returns dict:
    delta_n = E_e[n] - E_ref_exc
    in angular-frequency units [rad/s].
    """
    return {i: E_e[i] - E_ref_exc for i in excited_indices}


def effective_raman(
    Delta_avg,
    delta_two_photon,
    g_i,
    g_f,
    excited_indices,
    delta_n_dict,
    D_eig_phys,
    E1,
    E2,
):
    """
    Effective two-photon Raman frequency, summing over all excited states
    and all polarization channels:

    Omega_R = sum_n sum_{q1,q2} Omega_1n^(q1) conj(Omega_2n^(q2)) / [2 (Delta - delta_n)]

    where:
    Delta = (Delta1 + Delta2)/2     (average single-photon detuning)
    delta = Delta1 - Delta2          (two-photon detuning)

    Selection rules are encoded in the dipole matrix elements (zero when forbidden).

    Parameters
    ----------
    D_eig_phys : dict {q: matrix} with q in {-1, 0, +1}
        Dipole operator matrices in the eigenbasis, for each polarization q
    E1, E2 : dict {q: complex} for laser 1 and 2
        Complex electric field amplitude per polarization component [V/m]
    """
    Omega_R = 0.0 + 0.0j

    for ie in excited_indices:
        denom = Delta_avg - delta_n_dict[ie]
        for q1 in (-1, 0, +1):
            Om1 = -E1[q1] * D_eig_phys[q1][ie, g_i] / hbar
            for q2 in (-1, 0, +1):
                Om2 = -E2[q2] * D_eig_phys[q2][ie, g_f] / hbar
                Omega_R += Om1 * np.conj(Om2) / (2 * denom)

    return Omega_R

def ac_stark_shift(
    Delta_avg,
    ground_index,
    intended_laser,  # 1 or 2
    omega_21,
    excited_indices,
    delta_n_dict,
    D_eig_phys,
    E1,
    E2,
):
    """
    AC Stark shift including cross couplings.

    For a given ground state:
    - intended laser contributes with denominator (Delta - delta_n)
    - the other laser contributes with denominator shifted by ±omega_21

    Conventions:
    ground_index == g_i -> initial state
    ground_index == g_f -> final state

    intended_laser = 1 means this ground state is the one laser 1 is targeting
    intended_laser = 2 means this ground state is the one laser 2 is targeting

    Parameters
    ----------
    D_eig_phys : dict {q: matrix} with q in {-1, 0, +1}
    E1, E2 : dict {q: complex} for laser 1 and 2 [V/m]
    """
    shift = 0.0 + 0.0j

    if intended_laser == 1:
        E_int = E1
        E_cross = E2
        denom_cross_sign = +1  # laser 2 on g_i -> Delta - delta_n + omega_21
    elif intended_laser == 2:
        E_int = E2
        E_cross = E1
        denom_cross_sign = -1  # laser 1 on g_f -> Delta - delta_n - omega_21
    else:
        raise ValueError("intended_laser must be 1 or 2")

    for ie in excited_indices:
        denom_int = Delta_avg - delta_n_dict[ie]
        denom_cross = denom_int + denom_cross_sign * omega_21

        for q in (-1, 0, +1):
            Om_int = -E_int[q] * D_eig_phys[q][ie, ground_index] / hbar
            shift += abs(Om_int)**2 / (4 * denom_int)

            Om_cross = -E_cross[q] * D_eig_phys[q][ie, ground_index] / hbar
            shift += abs(Om_cross)**2 / (4 * denom_cross)

    return shift.real

# ============================================================
# Spontaneous scattering
# ============================================================

def scattering_rate(
    Delta_avg,
    ground_index,
    intended_laser,  # 1 for g_i, 2 for g_f
    omega_21,
    excited_indices,
    delta_n_dict,
    Gamma_e,
    D_eig_phys,
    E1,
    E2,
):
    """
    Estimate spontaneous scattering rate from one ground state.

    - intended laser uses denominator (Delta - delta_n)
    - cross laser uses denominator (Delta - delta_n ± omega_21)

    Sums over all polarization components q = -1, 0, +1.

    Parameters
    ----------
    D_eig_phys : dict {q: matrix} with q in {-1, 0, +1}
    E1, E2 : dict {q: complex} for laser 1 and 2 [V/m]
    """
    rate = 0.0

    if intended_laser == 1:
        E_int = E1
        E_cross = E2
        denom_cross_sign = +1.0  # laser 2 acting on g_i
    elif intended_laser == 2:
        E_int = E2
        E_cross = E1
        denom_cross_sign = -1.0  # laser 1 acting on g_f
    else:
        raise ValueError("intended_laser must be 1 or 2")

    for ie in excited_indices:
        Delta_int = Delta_avg - delta_n_dict[ie]
        Delta_cross = Delta_int + denom_cross_sign * omega_21

        for q in (-1, 0, +1):
            Om_int = -E_int[q] * D_eig_phys[q][ie, ground_index] / hbar
            rate += Gamma_e * abs(Om_int)**2 / (4 * Delta_int**2)

            Om_cross = -E_cross[q] * D_eig_phys[q][ie, ground_index] / hbar
            rate += Gamma_e * abs(Om_cross)**2 / (4 * Delta_cross**2)

    return rate

# ============================================================
# Evaluation & scanning functions
# ============================================================

def evaluate_parameter_set(
    I_per_beam_mWcm2,
    Delta_MHz,
    delta_two_photon_MHz,
    g_i,
    g_f,
    excited_indices,
    E_g,
    E_e,
    delta_n_dict,
    D_eig_phys,
    Gamma_e,
    omega_laser,
    beam1=None,
    beam2=None,
):
    """
    Evaluate one Raman parameter set and return a dictionary of useful quantities.

    Parameters
    ----------
    D_eig_phys  : dict {q: matrix} with q in {-1, 0, +1}
    omega_laser : float  Carrier angular frequency [rad/s], used for k_eff.
                          Pass the bare line frequency (line.omega); the
                          omega_1 vs omega_2 difference is at the 1e-7 level
                          and irrelevant for |k_eff| (see raman_recoil docstring).
    beam1, beam2 : Beam   Per-beam (theta_deg, alpha_deg, phi_deg). Default is
                          copropagating linear-45 (theta=0, alpha=45, phi=0)
                          on both, reproducing the old POL_LIN45 behaviour.
    """
    if beam1 is None:
        beam1 = Beam(alpha_deg=45)
    if beam2 is None:
        beam2 = Beam(alpha_deg=45)
    pol1 = beam1.pol()
    pol2 = beam2.pol()

    Delta_avg = 2 * np.pi * Delta_MHz * 1e6
    delta_two_photon = 2 * np.pi * delta_two_photon_MHz * 1e6

    E0 = electric_field_amplitude_from_intensity(I_per_beam_mWcm2)
    E1 = {q: pol1[q] * E0 for q in (-1, 0, +1)}
    E2 = {q: pol2[q] * E0 for q in (-1, 0, +1)}

    omega_21 = E_g[g_f] - E_g[g_i]

    Omega_R_eff = effective_raman(
        Delta_avg=Delta_avg,
        delta_two_photon=delta_two_photon,
        g_i=g_i, g_f=g_f,
        excited_indices=excited_indices,
        delta_n_dict=delta_n_dict,
        D_eig_phys=D_eig_phys,
        E1=E1, E2=E2,
    )

    omega_AC_i = ac_stark_shift(
        Delta_avg=Delta_avg,
        ground_index=g_i,
        intended_laser=1,
        omega_21=omega_21,
        excited_indices=excited_indices,
        delta_n_dict=delta_n_dict,
        D_eig_phys=D_eig_phys,
        E1=E1, E2=E2,
    )

    omega_AC_f = ac_stark_shift(
        Delta_avg=Delta_avg,
        ground_index=g_f,
        intended_laser=2,
        omega_21=omega_21,
        excited_indices=excited_indices,
        delta_n_dict=delta_n_dict,
        D_eig_phys=D_eig_phys,
        E1=E1, E2=E2,
    )

    Gamma_sc_i = scattering_rate(
        Delta_avg=Delta_avg,
        ground_index=g_i,
        intended_laser=1,
        omega_21=omega_21,
        excited_indices=excited_indices,
        delta_n_dict=delta_n_dict,
        Gamma_e=Gamma_e,
        D_eig_phys=D_eig_phys,
        E1=E1, E2=E2,
    )

    Gamma_sc_f = scattering_rate(
        Delta_avg=Delta_avg,
        ground_index=g_f,
        intended_laser=2,
        omega_21=omega_21,
        excited_indices=excited_indices,
        delta_n_dict=delta_n_dict,
        Gamma_e=Gamma_e,
        D_eig_phys=D_eig_phys,
        E1=E1, E2=E2,
    )

    Omega_R_abs = abs(Omega_R_eff)
    t_pi = np.pi / Omega_R_abs if Omega_R_abs > 0 else np.inf

    # AC Stark differential, kept under the original sign convention.
    delta_AC = -(omega_AC_f - omega_AC_i)

    # Photon-recoil shift. For atoms initially at rest the resonance
    # condition is omega_1 - omega_2 = omega_21 + (delta_AC contrib) + omega_rec,
    # so the AOM offset that hits resonance is delta_AC + omega_rec.
    # (k_eff . v_initial Doppler term is zero here; can be added later.)
    recoil = raman_recoil(beam1, beam2, omega_laser, mass=m_K39)
    omega_rec = recoil["omega_rec_2pi_kHz"] * 2 * np.pi * 1e3  # rad/s

    delta_req = delta_AC + omega_rec

    P_sc_i = Gamma_sc_i * t_pi if np.isfinite(t_pi) else np.inf
    P_sc_f = Gamma_sc_f * t_pi if np.isfinite(t_pi) else np.inf

    out = {
    "I_mWcm2": I_per_beam_mWcm2,
    "Delta_MHz": Delta_MHz,
    "delta_input_MHz": delta_two_photon_MHz,
    "Omega_R_2pi_MHz": Omega_R_eff.real / (2 * np.pi) / 1e6,
    "Omega_R_abs_2pi_MHz": Omega_R_abs / (2 * np.pi) / 1e6,
    "t_pi_us": t_pi * 1e6,
    "omega_AC_i_2pi_MHz": omega_AC_i / (2 * np.pi) / 1e6,
    "omega_AC_f_2pi_MHz": omega_AC_f / (2 * np.pi) / 1e6,
    "delta_AC_2pi_MHz": delta_AC / (2 * np.pi) / 1e6,
    "delta_req_2pi_MHz": delta_req / (2 * np.pi) / 1e6,
    "omega_21_2pi_MHz": omega_21 / (2 * np.pi) / 1e6,
    "Gamma_sc_i_2pi_kHz": Gamma_sc_i / (2 * np.pi) / 1e3,
    "Gamma_sc_f_2pi_kHz": Gamma_sc_f / (2 * np.pi) / 1e3,
    "P_sc_i": P_sc_i,
    "P_sc_f": P_sc_f,
    "P_sc_avg": (Gamma_sc_i + Gamma_sc_f) * t_pi / 2 if np.isfinite(t_pi) else np.inf,
    }
    out.update(recoil)
    return out


def print_parameter_report(res):
    """
    Pretty print one evaluated parameter set.
    """
    print("\nParameter report")
    print(f"I per Raman beam = {res['I_mWcm2']:.3f} mW/cm^2")
    print(f"Delta /2π = {res['Delta_MHz']:+.6f} MHz")
    print(f"delta input /2π = {res['delta_input_MHz']:+.6f} MHz")
    print(f"Omega_R /2π = {res['Omega_R_2pi_MHz']:+.6f} MHz")
    print(f"|Omega_R| /2π = {res['Omega_R_abs_2pi_MHz']:+.6f} MHz")
    print(f"t_pi = {res['t_pi_us']:.6f} us")
    print(f"omega_AC_i /2π = {res['omega_AC_i_2pi_MHz']:+.6f} MHz")
    print(f"omega_AC_f /2π = {res['omega_AC_f_2pi_MHz']:+.6f} MHz")
    print(f"delta_AC /2π   = {res['delta_AC_2pi_MHz']:+.6f} MHz")
    print(f"omega_rec /2π  = {res['omega_rec_2pi_kHz']:+.6f} kHz")
    print(f"delta_req /2π  = {res['delta_req_2pi_MHz']:+.6f} MHz  (= delta_AC + omega_rec)")
    print(f"omega_21 /2π = {res['omega_21_2pi_MHz']:+.6f} MHz")
    print(f"|k_eff| = {res['k_eff_per_um']:.6f} 1/um   v_rec = {res['v_rec_mm_s']:.4f} mm/s")
    print(f"Gamma_sc_i /2π = {res['Gamma_sc_i_2pi_kHz']:+.6f} kHz")
    print(f"Gamma_sc_f /2π = {res['Gamma_sc_f_2pi_kHz']:+.6f} kHz")
    print(f"P_sc_i ~ Gamma_i*t_pi = {res['P_sc_i']:.6e}")
    print(f"P_sc_f ~ Gamma_f*t_pi = {res['P_sc_f']:.6e}")
    print(f"P_sc_avg ~ (Gamma_i+Gamma_f)*t_pi/2 = {res['P_sc_avg']:.6e}")


def scan_parameter_grid(
    I_values_mWcm2,
    Delta_values_MHz,
    g_i,
    g_f,
    excited_indices,
    E_g,
    E_e,
    delta_n_dict,
    D_eig_phys,
    Gamma_e,
    omega_laser,
    beam1=None,
    beam2=None,
    target_tpi_us=1.0,
    avg_scatter_prob=None,
):
    """
    Scan a grid of (I, Delta), always first evaluating with delta_input = 0,
    then storing the Stark-corrected required delta.

    Returns a pandas DataFrame sorted by closeness to target pi time,
    optionally filtered by scattering probability.
    """
    rows = []

    for I_mWcm2 in I_values_mWcm2:
        for Delta_MHz in Delta_values_MHz:
            res = evaluate_parameter_set(
                I_per_beam_mWcm2=I_mWcm2,
                Delta_MHz=Delta_MHz,
                delta_two_photon_MHz=0.0,
                g_i=g_i, g_f=g_f,
                excited_indices=excited_indices,
                E_g=E_g, E_e=E_e,
                delta_n_dict=delta_n_dict,
                D_eig_phys=D_eig_phys,
                Gamma_e=Gamma_e,
                omega_laser=omega_laser,
                beam1=beam1, beam2=beam2,
            )
            rows.append(res)

    df = pd.DataFrame(rows)
    df["tpi_error_us"] = np.abs(df["t_pi_us"] - target_tpi_us)

    if avg_scatter_prob is not None:
        df = df[df["P_sc_avg"] <= avg_scatter_prob].copy()

    df = df.sort_values(["tpi_error_us", "P_sc_avg", "I_mWcm2"]).reset_index(drop=True)
    return df


def plot_intensity_vs_detuning(dfs_and_targets, label="", color_col="P_sc_avg"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (target, df) in zip(axes.flat, dfs_and_targets):
        best = df.loc[df.groupby("Delta_MHz")["tpi_error_us"].idxmin()]
        best = best.sort_values("Delta_MHz")
        sc = ax.scatter(
            best["Delta_MHz"],
            best["I_mWcm2"],
            c=best[color_col],
            cmap="viridis",
            vmin=0,
            vmax=1.0,
            s=15,
        )
        ax.set_xlabel("Detuning Δ (MHz)")
        ax.set_ylabel("Intensity I (mW/cm²)")
        ax.set_title(f"t_π ≈ {target} µs")
        fig.colorbar(sc, ax=ax, label=color_col)
    if label:
        fig.suptitle(label, fontsize=14, y=1.02)
    plt.tight_layout()
    filename = f"intensity_vs_detuning_{label}.png" if label else "intensity_vs_detuning.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_fixed_waist(df_all, label="", I_mWcm2=None):
    """
    Plot t_pi and P_sc_avg as a function of detuning for a fixed intensity.
    df_all is the full DataFrame from one scan (any target, they all have the same data).
    """
    df = df_all.sort_values("Delta_MHz")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(df["Delta_MHz"], df["t_pi_us"].clip(upper=100.0), ".-", markersize=3)
    ax1.set_ylabel("t_π (µs)")
    ax1.set_ylim(0, 100)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="t_π = 1 µs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["Delta_MHz"], df["P_sc_avg"].clip(upper=1.0), ".-", markersize=3)
    ax2.set_xlabel("Detuning Δ (MHz)")
    ax2.set_ylabel("P_sc_avg")
    ax2.set_ylim(0, 1.0)
    ax2.axhline(0.05, color="green", linestyle="--", alpha=0.5, label="5%")
    ax2.axhline(0.1, color="orange", linestyle="--", alpha=0.5, label="10%")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    title = label
    if I_mWcm2 is not None:
        title += f" (I = {I_mWcm2:.1f} mW/cm²)"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    filename = f"fixed_waist_{label}.png" if label else "fixed_waist.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Full analysis for a given (B, mF_i, mF_f) configuration
# ============================================================

def pol_elliptical_yz(alpha_deg, phi_deg=0):
    """
    Complex amplitude decomposition for elliptical polarization in the y-z plane.
    Propagation along x, quantization axis (B) along z.

    E = E0 (cos(alpha) z_hat + e^(i*phi) sin(alpha) y_hat)

    Decomposed in spherical basis:
        z_hat = e0,  y_hat = i(e+1 + e-1)/sqrt(2)

    Parameters
    ----------
    alpha_deg : float  Angle between polarization and B (controls pi/sigma balance)
    phi_deg : float    Phase between z and y components (controls ellipticity)

    Special cases:
        phi = 0       linear polarization at angle alpha
        phi = +90     right elliptical (circular when alpha = 45)
        phi = -90     left elliptical (circular when alpha = 45)
    """
    alpha = np.radians(alpha_deg)
    phi = np.radians(phi_deg)
    sigma_amp = np.exp(1j * phi) * np.sin(alpha) * 1j / np.sqrt(2)
    return {0: np.cos(alpha), +1: sigma_amp, -1: sigma_amp}


def pol_linear(theta_deg):
    """Linear polarization at angle theta to B. Shortcut for pol_elliptical_yz(theta, 0)."""
    return pol_elliptical_yz(theta_deg, phi_deg=0)


# Default polarization: linear at 45 deg to quantization axis
POL_LIN45 = pol_linear(45)


def pol_beam_xy(theta_deg, alpha_deg, phi_deg=0):
    """
    Polarization amplitudes for a beam propagating in the x-y plane at angle
    theta from x_hat, with quantization axis B along z_hat.

    The propagation direction is
        k_hat(theta) = cos(theta) x_hat + sin(theta) y_hat,
    so the in-plane direction perpendicular to k_hat is
        u_hat(theta) = -sin(theta) x_hat + cos(theta) y_hat.
    z_hat is always perpendicular to k_hat (the beam never propagates along B in
    this parametrisation), so the pi-component is available for any theta.

    Most general polarization compatible with k_hat(theta) (transverse + complex
    amplitude in the (z, u) plane):

        E/E0 = cos(alpha) z_hat + e^(i*phi) sin(alpha) u_hat(theta)

    Decomposing u_hat(theta) into the spherical basis:
        u_hat(theta) = (i/sqrt(2)) ( e^(-i theta) e_+1 + e^(+i theta) e_-1 ).
    The i/sqrt(2) factor matches the existing convention used by
    pol_elliptical_yz, so this function reduces to it at theta = 0.

    Parameters
    ----------
    theta_deg : float  Beam angle from x_hat in the x-y plane [degrees].
                       theta = 0   -> beam along +x  (matches pol_elliptical_yz).
                       theta = 90  -> beam along +y.
                       theta = 180 -> beam along -x  (counter-propagating).
    alpha_deg : float  Polarization tilt: cos(alpha) is the pi amplitude,
                       sin(alpha) is the in-plane (sigma+/sigma-) amplitude.
    phi_deg : float    Phase between the pi and the in-plane components.
                       phi = 0       linear polarization at angle alpha.
                       phi = +/-90   elliptical (circular when alpha = 45).

    Returns
    -------
    dict  {0: c_pi, +1: c_sigma_plus, -1: c_sigma_minus}, complex amplitudes
    in the spherical basis with the same convention as pol_elliptical_yz.

    Reduces exactly to pol_elliptical_yz(alpha, phi) when theta = 0.
    """
    alpha = np.radians(alpha_deg)
    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)

    # In-plane (sigma) amplitude before the theta-dependent phase factor.
    # Identical to the sigma_amp of pol_elliptical_yz when theta = 0.
    in_plane = np.exp(1j * phi) * np.sin(alpha) * 1j / np.sqrt(2)

    # Rotating k_hat in the x-y plane multiplies the sigma+ amplitude by
    # exp(-i theta) and the sigma- amplitude by exp(+i theta). The pi amplitude
    # is unaffected because z_hat is perpendicular to the rotation plane.
    sigma_plus  = in_plane * np.exp(-1j * theta)
    sigma_minus = in_plane * np.exp(+1j * theta)

    return {0: np.cos(alpha), +1: sigma_plus, -1: sigma_minus}


# ============================================================
# Beam = propagation direction + polarization
# ============================================================

@dataclass
class Beam:
    """
    A laser beam propagating in the x-y plane (perpendicular to B along z).

    Parameters
    ----------
    theta_deg : float  Propagation direction from +x in the x-y plane [deg].
                       theta = 0   -> beam along +x.
                       theta = 90  -> beam along +y.
                       theta = 180 -> beam along -x (counter-propagates beam at theta=0).
    alpha_deg : float  Polarization tilt: cos(alpha) is the pi amplitude (along z),
                       sin(alpha) is the in-plane (sigma+/sigma-) amplitude.
                       alpha = 0   -> pure pi (linear along B).
                       alpha = 90  -> pure sigma (linear perpendicular to B).
    phi_deg   : float  Phase between pi and in-plane components.
                       phi = 0       -> linear at angle alpha to B.
                       phi = +/-90   -> elliptical (circular when alpha = 45).
    """
    theta_deg: float = 0.0
    alpha_deg: float = 0.0
    phi_deg: float = 0.0

    def pol(self):
        """Spherical-basis polarization dict {q: complex} for q in {-1, 0, +1}."""
        return pol_beam_xy(self.theta_deg, self.alpha_deg, self.phi_deg)

    def k_hat(self):
        """Unit propagation vector in lab frame (B along z)."""
        t = np.radians(self.theta_deg)
        return np.array([np.cos(t), np.sin(t), 0.0])


def raman_recoil(beam1, beam2, omega_laser, mass=m_K39):
    """
    Two-photon Raman momentum kick and recoil quantities.

    The atom absorbs from beam 1 and emits into beam 2, so the momentum
    transfer is

        hbar k_eff = hbar (k_1 - k_2),
        with |k_1| = omega_1 / c, |k_2| = omega_2 / c.

    Strictly, omega_1 and omega_2 differ by the Raman drive frequency
    omega_1 - omega_2 = omega_21 + small corrections (~ MHz to GHz). Here
    we use a single carrier frequency omega_laser for both:

        |k_1| = |k_2| = omega_laser / c.

    Justification: the dominant scales are
        omega_laser ~ 2 pi x 390 THz,
        omega_21    ~ 2 pi x 100 MHz - 1.6 GHz,
        Delta       ~ 2 pi x 1 GHz at most.
    So |k_1| and |k_2| individually differ from omega_laser/c by at most
    ~Delta/omega_laser ~ 1e-6, and from each other by ~omega_21/omega_laser
    ~ 1e-7 - 1e-9. The resulting error on |k_eff| is well below any
    physically interesting level.

    For symmetric geometry (theta_1 = +theta, theta_2 = -theta) this gives
        k_eff . x_hat = (omega_1 - omega_2)/c * cos(theta)  (~ omega_21/c, tiny)
        k_eff . y_hat = (omega_1 + omega_2)/c * sin(theta)  (~ 2 omega_laser/c sin theta)
    So to part-per-million precision, |k_eff| = 2 (omega_laser/c) |sin theta|
    along y_hat.

    Pass omega_laser = line.omega (D1 or D2 bare line frequency) at the
    call site; the small Delta-dependence is absorbed in the approximation.

    Returns
    -------
    dict with:
      k_eff_x_per_um, k_eff_y_per_um : k_eff components [1/um]
      k_eff_per_um                   : |k_eff| [1/um]
      v_rec_mm_s                     : recoil velocity hbar |k_eff| / m [mm/s]
      omega_rec_2pi_kHz              : recoil shift hbar |k_eff|^2 / (2 m) [kHz, /2pi]
    """
    k_mag = omega_laser / c
    k_eff = k_mag * (beam1.k_hat() - beam2.k_hat())
    k_eff_norm = float(np.linalg.norm(k_eff))
    v_rec = hbar * k_eff_norm / mass
    omega_rec = hbar * k_eff_norm**2 / (2 * mass)
    return {
        "k_eff_x_per_um": float(k_eff[0]) * 1e-6,
        "k_eff_y_per_um": float(k_eff[1]) * 1e-6,
        "k_eff_per_um": k_eff_norm * 1e-6,
        "v_rec_mm_s": v_rec * 1e3,
        "omega_rec_2pi_kHz": omega_rec / (2 * np.pi) / 1e3,
    }


def run_analysis(B_G, mF_i, mF_f, label,
                 P_total_mW=5.0, L_cloud_m=30e-6,
                 target_psc_values=[0.05, 0.1],
                 report_Delta_MHz=None,
                 w0_fixed_mm=None,
                 beam1=None, beam2=None,
                 line=D1,
                 verbose=False):
    """
    Run the full Raman analysis for a given configuration.
    If w0_fixed_mm is set, the intensity is fixed by the beam waist and power,
    and the scan is only over detuning.

    Parameters
    ----------
    B_G : float         Magnetic field [Gauss]
    mF_i : float        Initial ground state mF
    mF_f : float        Final ground state mF
    label : str         Label for printouts and comparison table
    P_total_mW : float  Total laser power [mW], split 50/50
    L_cloud_m : float   Half-length of atom cloud [m]
    target_psc_values : list  Target P_sc_avg values for parameter reports
    beam1, beam2 : Beam Per-beam (theta_deg, alpha_deg, phi_deg). Default is
                        copropagating linear-45 (theta=0, alpha=45) on both,
                        reproducing the old behaviour.
    verbose : bool      Print detailed diagnostic output

    Returns
    -------
    list of dicts, one per target_psc, with parameter report + kick info
    """
    if beam1 is None:
        beam1 = Beam(alpha_deg=45)
    if beam2 is None:
        beam2 = Beam(alpha_deg=45)
    pol1 = beam1.pol()
    pol2 = beam2.pol()

    print(f"\n{'#'*60}")
    print(f"# {label}")
    print(f"# Line: {line.name}, B = {B_G} G, mF: {mF_i} -> {mF_f}")
    print(f"# beam1: theta={beam1.theta_deg} deg, alpha={beam1.alpha_deg} deg, phi={beam1.phi_deg} deg")
    print(f"# beam2: theta={beam2.theta_deg} deg, alpha={beam2.alpha_deg} deg, phi={beam2.phi_deg} deg")
    print(f"# pol1: pi={pol1[0]}, s+={pol1[+1]}, s-={pol1[-1]}")
    print(f"# pol2: pi={pol2[0]}, s+={pol2[+1]}, s-={pol2[-1]}")
    print(f"{'#'*60}")

    # ---------- Diagonalize ground S1/2 and excited manifold ----------
    ground_states, E_g, U_g, basis_g = diagonalize_manifold(
        J=0.5, I=C.I_nuclear, ahfs=C.ahfs_S, gJ=C.gJ_S, gI=C.gI,
        B_G=B_G, manifold_label="S1/2",
    )

    excited_states, E_e, U_e, basis_e = diagonalize_manifold(
        J=line.J_e, I=C.I_nuclear, ahfs=line.ahfs, gJ=line.gJ, gI=C.gI,
        B_G=B_G, manifold_label=f"P (J={line.J_e})", bhfs=line.bhfs,
    )

    save_states_to_json(ground_states, f"ground_states_{label}.json",
                        title=f"Ground manifold S1/2 at B = {B_G} G")
    save_states_to_json(excited_states, f"excited_states_{label}.json",
                        title=f"Excited manifold {line.name} (P, J={line.J_e}) at B = {B_G} G")

    if verbose:
        print_states(ground_states, title=f"Ground manifold S1/2 at B = {B_G} G")
        print_states(excited_states, title=f"Excited manifold {line.name} (J={line.J_e}) at B = {B_G} G")

    # ---------- Build unit dipole operators (for diagnostics) ----------
    if verbose:
        D_pi_unc = dipole_operator_uncoupled(
            basis_excited=basis_e, basis_ground=basis_g,
            Je=line.J_e, Jg=0.5, q=0, reduced_matrix_element=1.0,
        )
        D_sp_unc = dipole_operator_uncoupled(
            basis_excited=basis_e, basis_ground=basis_g,
            Je=line.J_e, Jg=0.5, q=+1, reduced_matrix_element=1.0,
        )
        D_sm_unc = dipole_operator_uncoupled(
            basis_excited=basis_e, basis_ground=basis_g,
            Je=line.J_e, Jg=0.5, q=-1, reduced_matrix_element=1.0,
        )
        print_nonzero_uncoupled_elements(D_pi_unc, basis_e, basis_g, "d_0")
        print_nonzero_uncoupled_elements(D_sp_unc, basis_e, basis_g, "d_+1")
        print_nonzero_uncoupled_elements(D_sm_unc, basis_e, basis_g, "d_-1")

        D_pi_eig = transform_operator_to_eigenbasis(D_pi_unc, U_e, U_g)
        D_sp_eig = transform_operator_to_eigenbasis(D_sp_unc, U_e, U_g)
        D_sm_eig = transform_operator_to_eigenbasis(D_sm_unc, U_e, U_g)

    # ---------- Identify useful states ----------
    g_mFi = select_states_by_mF(ground_states, target_mF=mF_i)
    g_mFf = select_states_by_mF(ground_states, target_mF=mF_f)

    if len(g_mFi) == 0 or len(g_mFf) == 0:
        raise RuntimeError(f"Could not find ground states with mF={mF_i} and mF={mF_f} at B={B_G} G.")

    g_i = g_mFi[0][0]
    g_f = g_mFf[0][0]

    # Excited states reachable from g_i (via q = -1, 0, +1)
    mF_from_gi = {mF_i + q for q in [-1, 0, +1]}
    # Excited states reachable from g_f (via q = -1, 0, +1)
    mF_from_gf = {mF_f + q for q in [-1, 0, +1]}
    # Only intermediates reachable from both ground states contribute
    mF_intermediates = sorted(mF_from_gi & mF_from_gf)

    e_by_mF = {}
    excited_indices = []
    for mF_target in mF_intermediates:
        sel = select_states_by_mF(excited_states, target_mF=mF_target)
        e_by_mF[mF_target] = sel
        for ie, _ in sel:
            excited_indices.append(ie)

    ground_indices = [g_i, g_f]

    print(f"  g_i = state {g_i} (mF~{mF_i}), g_f = state {g_f} (mF~{mF_f})")
    print(f"  Intermediate mF sectors: {mF_intermediates}")
    print(f"  Excited indices: {excited_indices}")

    if verbose:
        for mF_target in mF_intermediates:
            print(f"  Excited states with mF~{mF_target}: {[i for i, _ in e_by_mF[mF_target]]}")

        print_selected_eigen_couplings(D_pi_eig, excited_states, ground_states,
                                        excited_indices, ground_indices, "d_0")
        print_selected_eigen_couplings(D_sp_eig, excited_states, ground_states,
                                        excited_indices, ground_indices, "d_+1")
        print_selected_eigen_couplings(D_sm_eig, excited_states, ground_states,
                                        excited_indices, ground_indices, "d_-1")

        DeltamF = int(mF_f - mF_i)
        print(f"\n=== Raman path products for target DeltamF = {DeltamF} ===")
        q_names = {-1: "sigma-", 0: "pi", +1: "sigma+"}
        for mF_target, sel in e_by_mF.items():
            for ie, est in sel:
                for q_abs in [-1, 0, +1]:
                    q_emit = q_abs - DeltamF
                    if abs(q_emit) > 1:
                        continue
                    D_abs = {-1: D_sm_eig, 0: D_pi_eig, +1: D_sp_eig}[q_abs]
                    D_emt = {-1: D_sm_eig, 0: D_pi_eig, +1: D_sp_eig}[q_emit]
                    amp = D_emt[ie, g_f] * D_abs[ie, g_i]
                    if abs(amp) > 1e-12:
                        print(
                            f"  Path via e{ie} (mF~{mF_target}): "
                            f"absorb {q_names[q_abs]}, emit {q_names[q_emit]} -> "
                            f"product = {amp.real:+.6e}{amp.imag:+.6e}j"
                        )

    # ---------- Reduced dipole matrix element ----------
    d_red = reduced_dipole_from_linewidth(line.omega, line.gamma, Je=line.J_e)

    if verbose:
        print(f"  d_red ({line.name}) = {d_red:.6e} C*m")

    # ---------- Rebuild dipole operators with physical reduced matrix element ----------
    D_eig_phys = {}
    for q in (-1, 0, +1):
        D_unc_phys = dipole_operator_uncoupled(
            basis_excited=basis_e, basis_ground=basis_g,
            Je=line.J_e, Jg=0.5, q=q, reduced_matrix_element=d_red,
        )
        D_eig_phys[q] = transform_operator_to_eigenbasis(D_unc_phys, U_e, U_g)

    # ---------- Intensity grid setup ----------
    if w0_fixed_mm is not None:
        w0_m = w0_fixed_mm * 1e-3
        I_max = 2 * (P_total_mW / 2) * 1e-3 / (np.pi * w0_m**2) / 10  # -> mW/cm^2
        I_values = np.array([I_max])
        print(f"  Fixed w0 = {w0_fixed_mm} mm -> I_max = {I_max:.1f} mW/cm^2")
    else:
        I_values = np.linspace(10, 30000, 450)

    if verbose:
        E0_test = electric_field_amplitude_from_intensity(I_values[0])
        E1_test = {q: pol1[q] * E0_test for q in (-1, 0, +1)}
        E2_test = {q: pol2[q] * E0_test for q in (-1, 0, +1)}
        print(f"\nOne-photon Rabi frequencies at I = {I_values[0]:.1f} mW/cm^2")
        q_names = {-1: "sigma-", 0: "pi", +1: "sigma+"}
        for ie in excited_indices:
            print(f"\n  Excited state ie = {ie}, mF ~ {excited_states[ie].mF_expectation:+.1f}")
            for g_idx, g_label in [(g_i, "g_i"), (g_f, "g_f")]:
                for q in (-1, 0, +1):
                    Om1 = -E1_test[q] * D_eig_phys[q][ie, g_idx] / hbar
                    Om2 = -E2_test[q] * D_eig_phys[q][ie, g_idx] / hbar
                    print(f"   {q_names[q]:7s} beam1 (ie <- {g_label}) /2pi = {abs(Om1)/(2*np.pi)/1e6:+.6f} MHz")
                    print(f"   {q_names[q]:7s} beam2 (ie <- {g_label}) /2pi = {abs(Om2)/(2*np.pi)/1e6:+.6f} MHz")

    # ---------- Build reference excited-manifold structure ----------
    E_ref_exc = excited_manifold_reference_energy(E_e)
    delta_n_dict = excited_state_offsets(E_e, excited_indices, E_ref_exc)

    if verbose:
        print(f"\nReference excited-manifold energy E_ref_exc /2pi: {E_ref_exc/(2*np.pi)/1e6:+.6f} MHz")
        for ie in excited_indices:
            print(f"  state {ie}: delta_n /2pi = {delta_n_dict[ie]/(2*np.pi)/1e6:+.6f} MHz")

    # ---------- Grid scan ----------
    Delta_values = np.linspace(-3000, 3000, 300)

    target_tpi_values = [0.5, 1.0, 2.0, 3.0]
    dfs_and_targets = []
    for target in target_tpi_values:
        df = scan_parameter_grid(
            I_values_mWcm2=I_values,
            Delta_values_MHz=Delta_values,
            g_i=g_i, g_f=g_f,
            excited_indices=excited_indices,
            E_g=E_g, E_e=E_e,
            delta_n_dict=delta_n_dict,
            D_eig_phys=D_eig_phys,
            Gamma_e=line.gamma,
            omega_laser=line.omega,
            beam1=beam1, beam2=beam2,
            target_tpi_us=target,
        )
        dfs_and_targets.append((target, df))

    plot_label = label
    if w0_fixed_mm is not None:
        _, df_any = dfs_and_targets[0]
        plot_fixed_waist(df_any, label=plot_label, I_mWcm2=I_values[0])
    else:
        plot_intensity_vs_detuning(dfs_and_targets, label=plot_label)

    # ---------- Parameter reports for target P_sc_avg values ----------
    _, df_scan = dfs_and_targets[0]

    if w0_fixed_mm is not None:
        # Fixed waist: no t_pi constraint, find min detuning with P_sc_avg <= target
        df_filtered = df_scan.copy()
    else:
        # Free intensity: filter for t_pi ~ 1 us
        _, df_scan = dfs_and_targets[1]  # target = 1.0 us
        df_filtered = df_scan[df_scan["tpi_error_us"] < 0.1].copy()

    results = []
    for target_psc in target_psc_values:
        candidates = df_filtered[df_filtered["P_sc_avg"] <= target_psc]
        if len(candidates) == 0:
            constraint = "found" if w0_fixed_mm else "and t_pi ~ 1 us found"
            print(f"\nNo points with P_sc_avg <= {target_psc} {constraint}.")
            continue
        # Fixed waist: min detuning; free intensity: min intensity
        sort_col = "Delta_MHz" if w0_fixed_mm else "I_mWcm2"
        best_row = candidates.sort_values(sort_col).iloc[0]

        if w0_fixed_mm:
            print(f"\n{'='*60}")
            print(f"Min detuning with P_sc_avg <= {target_psc} (w0 = {w0_fixed_mm} mm)")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Min intensity with P_sc_avg <= {target_psc} for t_pi ~ 1 us")
            print(f"{'='*60}")
        res_best = evaluate_parameter_set(
            I_per_beam_mWcm2=best_row["I_mWcm2"],
            Delta_MHz=best_row["Delta_MHz"],
            delta_two_photon_MHz=0.0,
            g_i=g_i, g_f=g_f,
            excited_indices=excited_indices,
            E_g=E_g, E_e=E_e,
            delta_n_dict=delta_n_dict,
            D_eig_phys=D_eig_phys,
            Gamma_e=line.gamma,
            omega_laser=line.omega,
            beam1=beam1, beam2=beam2,
        )
        print_parameter_report(res_best)

        # Dipole kick from focused Gaussian beam
        if w0_fixed_mm is not None:
            w0 = w0_fixed_mm * 1e-3
        else:
            w0 = beam_waist_from_power(P_total_mW, res_best["I_mWcm2"])
        # Conservative: use the largest AC Stark shift of the two ground states.
        # Could also use the average (|omega_AC_i| + |omega_AC_f|) / 2,
        # since the atom spends comparable time in each state during the pi-pulse.
        omega_AC_max = max(abs(res_best["omega_AC_i_2pi_MHz"]), abs(res_best["omega_AC_f_2pi_MHz"])) * 2 * np.pi * 1e6
        kick = dipole_kick(omega_AC_max, w0, m_K39, L_cloud_m, res_best["t_pi_us"] * 1e-6)
        print_kick_report(kick)

        res_best["label"] = label
        res_best["B_G"] = B_G
        res_best["mF_i"] = mF_i
        res_best["mF_f"] = mF_f
        res_best.update(kick)
        results.append(res_best)

    # ---------- Optional: report at a specific detuning ----------
    if report_Delta_MHz is not None:
        I_report = I_values[0] if w0_fixed_mm is not None else I_values[len(I_values)//2]
        print(f"\n{'='*60}")
        print(f"Report at Delta/2pi = {report_Delta_MHz} MHz, I = {I_report:.1f} mW/cm^2")
        print(f"{'='*60}")
        res_fixed = evaluate_parameter_set(
            I_per_beam_mWcm2=I_report,
            Delta_MHz=report_Delta_MHz,
            delta_two_photon_MHz=0.0,
            g_i=g_i, g_f=g_f,
            excited_indices=excited_indices,
            E_g=E_g, E_e=E_e,
            delta_n_dict=delta_n_dict,
            D_eig_phys=D_eig_phys,
            Gamma_e=line.gamma,
            omega_laser=line.omega,
            beam1=beam1, beam2=beam2,
        )
        print_parameter_report(res_fixed)

        if w0_fixed_mm is not None:
            w0 = w0_fixed_mm * 1e-3
        else:
            w0 = beam_waist_from_power(P_total_mW, res_fixed["I_mWcm2"])
        omega_AC_max = max(abs(res_fixed["omega_AC_i_2pi_MHz"]), abs(res_fixed["omega_AC_f_2pi_MHz"])) * 2 * np.pi * 1e6
        kick = dipole_kick(omega_AC_max, w0, m_K39, L_cloud_m, res_fixed["t_pi_us"] * 1e-6)
        print_kick_report(kick)

        res_fixed["label"] = label + f"_Delta{report_Delta_MHz}"
        res_fixed["B_G"] = B_G
        res_fixed["mF_i"] = mF_i
        res_fixed["mF_f"] = mF_f
        res_fixed.update(kick)
        results.append(res_fixed)

    return results


# ============================================================
# Polarisation angle scan
# ============================================================

def scan_polarization_angle(B_G, mF_i, mF_f, Delta_MHz,
                            I_mWcm2=None,
                            P_total_mW=5.0, w0_fixed_mm=None,
                            alphas_deg=np.arange(5, 86, 5),
                            theta1_deg=0.0, theta2_deg=0.0,
                            line=D1,
                            label=""):
    """
    Scan the linear polarisation angle alpha (shared by both beams) at a fixed
    (I, Delta), and plot t_pi and P_sc_avg vs alpha.

    Parameters
    ----------
    B_G : float          Magnetic field [Gauss]
    mF_i, mF_f : float   Initial and final mF
    Delta_MHz : float    Single-photon detuning [MHz]
    I_mWcm2 : float      Intensity per beam [mW/cm^2] (if None, computed from w0 and P)
    P_total_mW : float   Total laser power [mW], split 50/50
    w0_fixed_mm : float  Beam waist [mm] (used to compute I if I_mWcm2 is None)
    alphas_deg : array   Polarisation angles to scan [deg] (between E and B)
    theta1_deg : float   Propagation direction of beam 1 in xy-plane [deg]
    theta2_deg : float   Propagation direction of beam 2 in xy-plane [deg]
                         Default (0, 0) = copropagating along +x.
    line : LineConfig    Optical line (D1 or D2)
    label : str          Label for plot title and filename
    """
    if I_mWcm2 is None:
        if w0_fixed_mm is None:
            raise ValueError("Either I_mWcm2 or w0_fixed_mm must be provided")
        w0_m = w0_fixed_mm * 1e-3
        I_mWcm2 = 2 * (P_total_mW / 2) * 1e-3 / (np.pi * w0_m**2) / 10
    # Diagonalize
    ground_states, E_g, U_g, basis_g = diagonalize_manifold(
        J=0.5, I=C.I_nuclear, ahfs=C.ahfs_S, gJ=C.gJ_S, gI=C.gI,
        B_G=B_G, manifold_label="S1/2")
    excited_states, E_e, U_e, basis_e = diagonalize_manifold(
        J=line.J_e, I=C.I_nuclear, ahfs=line.ahfs, gJ=line.gJ, gI=C.gI,
        B_G=B_G, manifold_label=f"P (J={line.J_e})", bhfs=line.bhfs)

    # Dipole operators with physical reduced matrix element
    d_red = reduced_dipole_from_linewidth(line.omega, line.gamma, Je=line.J_e)
    D_eig_phys = {}
    for q in (-1, 0, +1):
        D_unc = dipole_operator_uncoupled(
            basis_e, basis_g, Je=line.J_e, Jg=0.5, q=q, reduced_matrix_element=d_red)
        D_eig_phys[q] = transform_operator_to_eigenbasis(D_unc, U_e, U_g)

    # Ground and excited state selection
    g_i = select_states_by_mF(ground_states, target_mF=mF_i)[0][0]
    g_f = select_states_by_mF(ground_states, target_mF=mF_f)[0][0]

    mF_from_gi = {mF_i + q for q in [-1, 0, +1]}
    mF_from_gf = {mF_f + q for q in [-1, 0, +1]}
    mF_intermediates = sorted(mF_from_gi & mF_from_gf)
    excited_indices = []
    for mF_target in mF_intermediates:
        for ie, _ in select_states_by_mF(excited_states, target_mF=mF_target):
            excited_indices.append(ie)

    E_ref_exc = excited_manifold_reference_energy(E_e)
    delta_n_dict = excited_state_offsets(E_e, excited_indices, E_ref_exc)

    # Scan
    rows = []
    for alpha in alphas_deg:
        beam1 = Beam(theta_deg=theta1_deg, alpha_deg=alpha, phi_deg=0.0)
        beam2 = Beam(theta_deg=theta2_deg, alpha_deg=alpha, phi_deg=0.0)
        res = evaluate_parameter_set(
            I_per_beam_mWcm2=I_mWcm2, Delta_MHz=Delta_MHz,
            delta_two_photon_MHz=0.0,
            g_i=g_i, g_f=g_f,
            excited_indices=excited_indices,
            E_g=E_g, E_e=E_e,
            delta_n_dict=delta_n_dict,
            D_eig_phys=D_eig_phys,
            Gamma_e=line.gamma,
            omega_laser=line.omega,
            beam1=beam1, beam2=beam2,
        )
        res["alpha_deg"] = alpha
        rows.append(res)

    df = pd.DataFrame(rows)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Polarisation angle scan: {label}")
    print(f"B = {B_G} G, mF = {mF_i} -> {mF_f}")
    print(f"Delta/2pi = {Delta_MHz} MHz, I = {I_mWcm2:.1f} mW/cm^2")
    print(f"theta1 = {theta1_deg} deg, theta2 = {theta2_deg} deg")
    print(f"{'='*60}")
    print(df[["alpha_deg", "Omega_R_abs_2pi_MHz", "t_pi_us", "P_sc_avg",
              "omega_AC_i_2pi_MHz", "omega_AC_f_2pi_MHz"]].to_string(index=False))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(df["alpha_deg"], df["t_pi_us"], "o-")
    ax1.set_ylabel("t_π (µs)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["alpha_deg"], df["P_sc_avg"].clip(upper=1.0), "o-")
    ax2.set_xlabel("Polarisation angle α (deg)")
    ax2.set_ylabel("P_sc_avg")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

    title = f"B = {B_G} G, mF = {mF_i} -> {mF_f}, Δ/2π = {Delta_MHz} MHz"
    fig.suptitle(title)
    plt.tight_layout()
    filename = f"pol_angle_scan_{label}.png" if label else "pol_angle_scan.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return df


# ============================================================
# Elliptical polarisation 2D scan (alpha, phi)
# ============================================================

def scan_polarization_ellipse(B_G, mF_i, mF_f, Delta_MHz,
                              I_mWcm2=None,
                              P_total_mW=5.0, w0_fixed_mm=None,
                              alphas_deg=np.arange(5, 86, 5),
                              phis_deg=np.arange(-90, 91, 5),
                              theta1_deg=0.0, theta2_deg=0.0,
                              line=D1,
                              label=""):
    """
    Scan elliptical polarisation (same on both beams) over a 2D grid (alpha, phi).
    Plots t_pi and P_sc_avg as heatmaps.

    Parameters
    ----------
    alphas_deg : array   Angle between polarization major axis and B [degrees]
    phis_deg : array     Phase between z and y components [degrees]
                         (0 = linear, +/-90 = circular)
    theta1_deg : float   Propagation direction of beam 1 in xy-plane [deg]
    theta2_deg : float   Propagation direction of beam 2 in xy-plane [deg]
                         Default (0, 0) = copropagating along +x.
    line : LineConfig    Optical line (D1 or D2)
    """
    if I_mWcm2 is None:
        if w0_fixed_mm is None:
            raise ValueError("Either I_mWcm2 or w0_fixed_mm must be provided")
        w0_m = w0_fixed_mm * 1e-3
        I_mWcm2 = 2 * (P_total_mW / 2) * 1e-3 / (np.pi * w0_m**2) / 10

    # Diagonalize
    ground_states, E_g, U_g, basis_g = diagonalize_manifold(
        J=0.5, I=C.I_nuclear, ahfs=C.ahfs_S, gJ=C.gJ_S, gI=C.gI,
        B_G=B_G, manifold_label="S1/2")
    excited_states, E_e, U_e, basis_e = diagonalize_manifold(
        J=line.J_e, I=C.I_nuclear, ahfs=line.ahfs, gJ=line.gJ, gI=C.gI,
        B_G=B_G, manifold_label=f"P (J={line.J_e})", bhfs=line.bhfs)

    d_red = reduced_dipole_from_linewidth(line.omega, line.gamma, Je=line.J_e)
    D_eig_phys = {}
    for q in (-1, 0, +1):
        D_unc = dipole_operator_uncoupled(
            basis_e, basis_g, Je=line.J_e, Jg=0.5, q=q, reduced_matrix_element=d_red)
        D_eig_phys[q] = transform_operator_to_eigenbasis(D_unc, U_e, U_g)

    g_i = select_states_by_mF(ground_states, target_mF=mF_i)[0][0]
    g_f = select_states_by_mF(ground_states, target_mF=mF_f)[0][0]

    mF_from_gi = {mF_i + q for q in [-1, 0, +1]}
    mF_from_gf = {mF_f + q for q in [-1, 0, +1]}
    mF_intermediates = sorted(mF_from_gi & mF_from_gf)
    excited_indices = []
    for mF_target in mF_intermediates:
        for ie, _ in select_states_by_mF(excited_states, target_mF=mF_target):
            excited_indices.append(ie)

    E_ref_exc = excited_manifold_reference_energy(E_e)
    delta_n_dict = excited_state_offsets(E_e, excited_indices, E_ref_exc)

    # 2D scan
    t_pi_grid = np.zeros((len(alphas_deg), len(phis_deg)))
    P_sc_grid = np.zeros((len(alphas_deg), len(phis_deg)))

    for i, alpha in enumerate(alphas_deg):
        for j, phi in enumerate(phis_deg):
            beam1 = Beam(theta_deg=theta1_deg, alpha_deg=alpha, phi_deg=phi)
            beam2 = Beam(theta_deg=theta2_deg, alpha_deg=alpha, phi_deg=phi)
            res = evaluate_parameter_set(
                I_per_beam_mWcm2=I_mWcm2, Delta_MHz=Delta_MHz,
                delta_two_photon_MHz=0.0,
                g_i=g_i, g_f=g_f,
                excited_indices=excited_indices,
                E_g=E_g, E_e=E_e,
                delta_n_dict=delta_n_dict,
                D_eig_phys=D_eig_phys,
                Gamma_e=line.gamma,
                omega_laser=line.omega,
                beam1=beam1, beam2=beam2,
            )
            t_pi_grid[i, j] = res["t_pi_us"]
            P_sc_grid[i, j] = res["P_sc_avg"]

    # Find minimum P_sc
    idx_min = np.unravel_index(np.argmin(P_sc_grid), P_sc_grid.shape)
    print(f"\n{'='*60}")
    print(f"Elliptical polarisation scan: {label}")
    print(f"B = {B_G} G, mF = {mF_i} -> {mF_f}, Delta/2pi = {Delta_MHz} MHz")
    print(f"I = {I_mWcm2:.1f} mW/cm^2")
    print(f"{'='*60}")
    print(f"Min P_sc_avg = {P_sc_grid[idx_min]:.4f} at alpha = {alphas_deg[idx_min[0]]} deg, phi = {phis_deg[idx_min[1]]} deg")
    print(f"  (t_pi at that point = {t_pi_grid[idx_min]:.3f} us)")

    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    extent = [phis_deg[0], phis_deg[-1], alphas_deg[0], alphas_deg[-1]]

    im1 = ax1.imshow(t_pi_grid.clip(max=100), extent=extent, origin="lower",
                      aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax1.set_xlabel("Phase phi (deg)")
    ax1.set_ylabel("Angle alpha (deg)")
    ax1.set_title("t_pi (us)")
    fig.colorbar(im1, ax=ax1, label="t_pi (us)")

    im2 = ax2.imshow(P_sc_grid.clip(max=1.0), extent=extent, origin="lower",
                      aspect="auto", cmap="viridis", vmin=0, vmax=1.0)
    ax2.set_xlabel("Phase phi (deg)")
    ax2.set_ylabel("Angle alpha (deg)")
    ax2.set_title("P_sc_avg")
    fig.colorbar(im2, ax=ax2, label="P_sc_avg")

    title = f"B = {B_G} G, mF = {mF_i} -> {mF_f}, Δ/2π = {Delta_MHz} MHz, I = {I_mWcm2:.0f} mW/cm²"
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    filename = f"pol_ellipse_scan_{label}.png" if label else "pol_ellipse_scan.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return alphas_deg, phis_deg, t_pi_grid, P_sc_grid


# ============================================================
# This file is a library. To run analyses, use:
#   python run_D1.py   (D1 line)
#   python run_D2.py   (D2 line)
# ============================================================
