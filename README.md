# Interacting particle simulations

Python simulations of stochastic particle systems.

## Birth-death chemotaxis

The [`birth_death_chemotaxis/`](birth_death_chemotaxis/) directory contains a
two-dimensional model in which cells emit chemical particles. Chemicals
diffuse and die, while cells move in response to the resulting chemical
measure. It includes a paired grid-based chemical-field approximation,
empirical cell Wasserstein comparisons, repeatable runs, a particle-count
sweep, and tests.

To run it:

```bash
cd birth_death_chemotaxis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_simulation.py
```

See the [simulation README](birth_death_chemotaxis/README.md) for the model,
parameters, numerical choices, and other commands.

## Earlier Brownian-motion script

[`ParticleBM2D.py`](ParticleBM2D.py) is the repository's original simple
two-dimensional Brownian-motion animation. It remains available as a separate
legacy example; its dependencies are separate from the chemotaxis simulator.
