# Birth-death chemotaxis simulations

This is a rebuilt Python simulation of a two-species chemical-particle
process from an undergraduate project on interacting particle systems. It
runs two coupled descriptions side by side:

1. **Individual chemicals:** each chemical diffuses independently, is born
   at each biological particle at rate `c_b`, and dies at rate `c_d`.
2. **Chemical field:** a periodic grid carries an approximation to chemical mass,
   diffuses it with coefficient `alpha_c²/2`, receives births from its own
   particle system at the same rate, and loses mass at rate `c_d`.

Both particle systems start at the same locations and receive the same
Brownian increments. This makes their torus root mean squared (RMS) distance
a useful pathwise comparison. One pathwise comparison alone is **not** a
propagation-of-chaos result.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_simulation.py
```

The default run uses `N=50`, `dt=0.002`, `450` steps (`0.9` time units),
`alpha_p=0.5`, `alpha_c=0.4`, `c_b=8`, `c_d=2`, and a `100 × 100` grid, as in
Table 5.1 and the shorter example in Chapter 5. The program writes:

- `output/birth_death_simulation.png`: final positions, field, pathwise RMS,
  and chemical count/mass balance;
- `output/birth_death_metrics.csv`: one row per time step;
- `output/birth_death_simulation.npz`: all metrics and sampled states. The
  chemical snapshots are concatenated in `chemical_positions`; snapshot `k`
  is the slice from `chemical_offsets[k]` to `chemical_offsets[k+1]`.

To run for 4 seconds (the duration used in the particle-count comparison):

```bash
python run_simulation.py --steps 2000 --seed 2023
```

Add `--gif` to save a side-by-side animation from the sampled states. Use
`--snapshot-stride 10` for a smoother animation (and a larger output file).

To reproduce the appendix's *initialization* choice more closely, use
`--initial-chemicals one_uniform`. The default instead places one initial
chemical at each particle, as stated in Chapter 4. `--initial-particles
single` places all particles uniformly in `[0,0.25]²`, as described in
Chapter 5; the default `split` uses the appendix's two squares. The other
parameters can be changed with `python run_simulation.py --help`.

NumPy alone suffices for data generation: use `--no-plot` if Matplotlib is
unavailable. Run the standard-library checks with:

```bash
python -m unittest discover -s tests -v
```

For the Chapter 5 particle-count comparison, run repeated seeds and save the
mean, sample standard deviation, and standard error of the **final RMS**:

```bash
python run_sweep.py --counts 20,50,100,200,400 --repeats 5 --steps 2000
```

The sweep writes a CSV summary, an NPZ file of every seed's RMS time series,
and a comparison plot under `output/sweep/`. A full five-count, five-seed,
four-second sweep may take several minutes; use smaller `--counts` or
`--steps` for a quick check. Error bars are standard errors across seeds.

## Implemented equations and numerical choices

On the unit torus, with chemical locations `C_j` and mass `m` per chemical,
the stochastic particles follow

```text
dX_i = alpha_p dB_i + m sum_j kappa(C_j, X_i) dt.
```

The local repulsive kernel follows the intent of Appendix A:

```text
kappa(c, x) = minimum_image(x - c)
              if ||minimum_image(x - c)||_infinity <= 0.25,
              else 0.
```

The cutoff and torus displacement are applied consistently in both models.
The appendix did not wrap the force displacement correctly and gave a
spurious force when no chemical was nearby; both are corrected here.

Particles and chemicals use Gaussian increments proportional to `sqrt(dt)`.
Chemical births are Poisson events with rate `c_b` **per particle**. Existing
chemicals survive a step with probability `exp(-c_d dt)`. Births have uniform
event times within the step and diffuse/survive over their remaining time.
Particle positions are frozen as birth sources over each step, giving an
Euler approximation to their continuous motion.

The field stores **cell masses**, not density values. Source mass per unit
time is `m c_b` at each field-driven particle, deposited on neighboring cell
centers. The grid heat/death equation with frozen source is solved exactly
for each time step in Fourier space using the discrete periodic Laplacian.
This avoids the instability of the appendix's explicit diffusion update at
its stated grid spacing and time step. The plot converts cell mass to density
by dividing by the cell area. The force uses the Appendix-inspired kernel
sampled at grid centers and bilinearly interpolated to particle locations.
Near the hard cutoff boundary, that interpolation smooths the force over
roughly one grid cell. Paired-path differences therefore include **grid
error** as well as chemical birth/death noise; a grid-refinement study is
needed before interpreting their size as only stochastic error.

## Report inconsistencies that matter for thesis work

- Table 5.1 sets `m = 2/(c_b N)`. With births at rate `c_b` per particle,
  the field source is then `m c_b N μ_N = 2 μ_N`, where `μ_N` is the
  normalized particle empirical measure. Equation (4.2) instead prints
  `c_b law(X) = 8 law(X)` for the table's `c_b=8`. This implementation uses
  the table's mass and the physically corresponding source in **both**
  simulations. Set `--chemical-mass 0.02` for `N=50` if the desired source
  is `8 μ_N` (`m=1/N`).
- The appendix's numerical values differ from Table 5.1 (`N=10` and
  `alpha_c=1` in the code). The defaults here use the table's values.
- The hard radius cutoff in the appendix's kernel is discontinuous, so it
  does not satisfy the Lipschitz assumption in Chapter 4's well-posedness
  theorem. A smooth kernel would be appropriate for a theory-aligned version.
- The field uses a finite grid and sources from its own finite particle
  population. It is a numerical surrogate for the chemical process, not the
  conditional expectation of the explicit system and not a direct simulation
  of the limiting law in Equation (4.2).

For fixed particle number `N` and `c_d > 0`, the expected chemical count is
`C(0)e^(-c_d t) + (N c_b/c_d)(1-e^(-c_d t))`; the field's total mass divided
by `m` follows the same identity. This is shown in the generated plot.
