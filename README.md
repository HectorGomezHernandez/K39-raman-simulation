# K-39 Stimulated Raman π-pulse Simulation

Numerical study of stimulated two-photon Raman transitions on the **potassium-39
D1 (4²S₁/₂ → 4²P₁/₂)** and **D2 (4²S₁/₂ → 4²P₃/₂)** lines in the
**Paschen–Back regime** (strong magnetic field), for a pair of non-copropagating
Raman beams.

For a given transition, magnetic field, laser detuning Δ, intensity and beam
geometry the code computes the quantities that matter for driving a clean
π-pulse:

- **t_π** — the π-pulse duration,
- **Ω_R** — the effective two-photon Rabi frequency,
- **P_sc** — the single-photon (spontaneous-scattering) loss probability,
- **δ_AC** — the differential AC-Stark shift,
- the two-photon recoil, and the Δ needed to keep P_sc below a target.

The atomic structure is treated exactly by diagonalizing the full hyperfine +
Zeeman Hamiltonian of each manifold; the Raman coupling is then built from the
dipole matrix elements between the resulting eigenstates, summed over **all**
intermediate excited states (no three-level approximation).

## Physics setup

- **Lines:** D1 (P₁/₂) and D2 (P₃/₂).
- **Magnetic fields:** 402.7 G, 445.4 G and 526 G (Paschen–Back regime).
- **Transitions studied:** mF = 1→0, 1→2, 0→1, −1→0 (ground 4²S₁/₂ manifold).
- **Polarization geometries:** linear-45°, π+σ, σ+π, circular (R+R), plus
  polarization-angle and polarization-ellipse scans.
- **Beam geometry:** two beams in the *xy*-plane, symmetric about *x̂*, separated
  by an angle that is scanned over **0°, 66.4°, 133.6° and 180°**.
  133.6° corresponds to the experimental apparatus; 0° is copropagating
  (no two-photon recoil) and 180° is fully counter-propagating.
- **Reference condition:** fixed waist w₀ = 1 mm, I = 159.155 mW/cm² (P = 5 mW
  per beam, 50/50 split).

## Repository layout

```
Raman_simulation.py        Core engine: Hamiltonians, diagonalization, dipole
                           operators, Raman coupling, π-pulse / P_sc / AC-Stark.
constants.json             K-39 atomic constants (hyperfine, g-factors, linewidths).

run_D1.py                  Generate the D1 result set for a given beam separation.
run_D2.py                  Generate the D2 result set.
intensity_scan.py          Scan intensity → smallest Δ keeping P_sc below 5 %/10 %.
report_helper.py           Fixed-waist summary table (Δ at P_sc = 5 %/10 %, t_π…).
tests.py                   Regression + physics sanity checks (sum rules, norms).
sanity_checks/             Old (filtered-intermediate) vs new (full-manifold)
                           reference comparison, with figures.

results/
  D1/<sep>deg/             D1 results per beam separation (0.0, 66.4, 133.6, 180.0).
  D2/                      D2 results (experimental geometry).

report_summary.{csv,txt}            Fixed-waist summary across geometries.
intensity_scan_summary.{csv,txt}    Intensity-scan summary.
```

Each `results/` folder contains, per (transition × polarization):

- `ground_states_*.json`, `excited_states_*.json` — eigenstate energies,
  ⟨mF⟩ and dominant basis components from the Hamiltonian diagonalization;
- `fixed_waist_*.png` — t_π and P_sc vs detuning at w₀ = 1 mm;
- `intensity_vs_detuning_*.png` — free-waist intensity/detuning behaviour;
- `pol_angle_scan_*.png`, `pol_ellipse_scan_*.png` — polarization scans;
- `comparison_table.txt` — all scalar outputs in one table.

## Requirements

Python 3 with `numpy`, `scipy`, `sympy`, `pandas`, `matplotlib`:

```bash
pip install numpy scipy sympy pandas matplotlib
```

## Usage

```bash
# D1, for one or several beam separations (degrees between propagation directions)
python run_D1.py 133.6      # experimental geometry
python run_D1.py 0          # copropagating
python run_D1.py 180        # counter-propagating

# D2 (experimental geometry)
python run_D2.py

# Summary tables (written next to the scripts)
python report_helper.py
python intensity_scan.py

# Checks
python tests.py
python sanity_checks/compare_old_vs_new_reference.py
```

`run_D1.py` writes to `results/D1/<sep>deg/` and `run_D2.py` to `results/D2/`.
The committed `results/` were produced with these scripts and are included so the
figures and eigenstate data can be browsed without rerunning anything.

## License

MIT — see [LICENSE](LICENSE).
