"""Coupled particle/chemical and chemical-field simulations on the unit torus.

The discrete model is adapted from an undergraduate project report:

    dX_i = alpha_p dB_i + m sum_j kappa(C_j, X_i) dt,

with independent chemical diffusion, births at each particle at rate ``birth_rate``,
and independent chemical deaths at rate ``death_rate``. The field model evolves
a grid approximation to chemical *cell mass* under the same source and death
scaling. Both particle systems use the same Brownian increments for a pathwise
comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def minimum_image(displacement: Array) -> Array:
    """Shortest signed displacement in each coordinate of the unit torus."""
    return (np.asarray(displacement) + 0.5) % 1.0 - 0.5


def torus_rms(first: Array, second: Array) -> float:
    """Root mean squared paired Euclidean distance on the unit torus."""
    delta = minimum_image(first - second)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


@dataclass(frozen=True)
class SimulationConfig:
    n_particles: int = 50
    steps: int = 450
    dt: float = 0.002
    alpha_particle: float = 0.5
    alpha_chemical: float = 0.4
    birth_rate: float = 8.0
    death_rate: float = 2.0
    chemical_mass: float | None = None
    interaction_radius: float = 0.25
    grid_size: int = 100
    snapshot_stride: int = 50
    seed: int = 2023
    initial_particles: str = "split"  # appendix: two uniform squares
    initial_chemicals: str = "at_particles"  # Chapter 4: one per particle

    def __post_init__(self) -> None:
        if self.n_particles < 1 or self.steps < 1 or self.grid_size < 4:
            raise ValueError("n_particles and steps must be positive; grid_size >= 4")
        if self.snapshot_stride < 1 or self.dt <= 0:
            raise ValueError("snapshot_stride and dt must be positive")
        if self.alpha_particle < 0 or self.alpha_chemical < 0:
            raise ValueError("diffusion coefficients cannot be negative")
        if self.birth_rate < 0 or self.death_rate < 0:
            raise ValueError("birth and death rates cannot be negative")
        if self.chemical_mass is not None and self.chemical_mass <= 0:
            raise ValueError("chemical_mass must be positive")
        if not 0 < self.interaction_radius < 0.5:
            raise ValueError("interaction_radius must be between 0 and 0.5")
        if self.initial_particles not in {"split", "single"}:
            raise ValueError("initial_particles must be 'split' or 'single'")
        if self.initial_chemicals not in {"at_particles", "one_uniform", "none"}:
            raise ValueError("invalid initial_chemicals value")

    @property
    def mass(self) -> float:
        if self.chemical_mass is not None:
            return self.chemical_mass
        if self.birth_rate == 0:
            return 1.0 / self.n_particles
        # Table 5.1 of the report, not the inconsistent source in Eq. (4.2).
        return 2.0 / (self.birth_rate * self.n_particles)


def initial_positions(config: SimulationConfig, rng: np.random.Generator) -> tuple[Array, Array]:
    """Draw initial particles, then initial chemicals under the selected convention."""
    n = config.n_particles
    if config.initial_particles == "single":
        particles = 0.25 * rng.random((n, 2))
    else:
        first_count = n // 2
        second_count = n - first_count
        particles = np.concatenate(
            (
                0.25 * rng.random((first_count, 2)),
                0.15 + 0.25 * rng.random((second_count, 2)),
            ),
            axis=0,
        )
    if config.initial_chemicals == "at_particles":
        chemicals = particles.copy()
    elif config.initial_chemicals == "one_uniform":
        chemicals = rng.random((1, 2))
    else:
        chemicals = np.empty((0, 2), dtype=float)
    return particles, chemicals


def chemical_force(
    particles: Array, chemicals: Array, mass: float, radius: float
) -> Array:
    """Repulsive local force from Appendix A, with correct periodic displacement.

    kappa(c, x) = minimum_image(x-c) when its infinity norm is <= radius, and
    zero otherwise.  The empty chemical set gives zero force.
    """
    if chemicals.shape[0] == 0:
        return np.zeros_like(particles)
    result = np.zeros_like(particles)
    # Chunking limits temporary memory for the larger N used in Figure 5.3.
    for start in range(0, particles.shape[0], 128):
        delta = minimum_image(particles[start : start + 128, None, :] - chemicals[None, :, :])
        nearby = np.all(np.abs(delta) <= radius, axis=2)
        result[start : start + 128] = mass * np.sum(delta * nearby[:, :, None], axis=1)
    return result


def deposit_cell_mass(positions: Array, weights: Array, grid_size: int) -> Array:
    """Periodically deposit masses at four neighboring cell centers (CIC)."""
    result = np.zeros((grid_size, grid_size), dtype=float)
    if positions.shape[0] == 0:
        return result
    positions = np.asarray(positions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or weights.shape != (len(positions),):
        raise ValueError("positions must be (n, 2), weights must be (n,)")
    scaled = (positions % 1.0) * grid_size - 0.5
    base = np.floor(scaled).astype(np.int64)
    fraction = scaled - base
    for offset_x in (0, 1):
        for offset_y in (0, 1):
            wx = fraction[:, 0] if offset_x else 1.0 - fraction[:, 0]
            wy = fraction[:, 1] if offset_y else 1.0 - fraction[:, 1]
            np.add.at(
                result,
                ((base[:, 0] + offset_x) % grid_size, (base[:, 1] + offset_y) % grid_size),
                weights * wx * wy,
            )
    return result


def _sample_grid_center_values(values: Array, positions: Array) -> Array:
    """Bilinearly interpolate a scalar grid whose indices denote cell centers."""
    size = values.shape[0]
    scaled = (positions % 1.0) * size - 0.5
    base = np.floor(scaled).astype(np.int64)
    fraction = scaled - base
    x0, y0 = base[:, 0] % size, base[:, 1] % size
    x1, y1 = (x0 + 1) % size, (y0 + 1) % size
    fx, fy = fraction[:, 0], fraction[:, 1]
    return (
        values[x0, y0] * (1 - fx) * (1 - fy)
        + values[x1, y0] * fx * (1 - fy)
        + values[x0, y1] * (1 - fx) * fy
        + values[x1, y1] * fx * fy
    )


class ChemicalField:
    """Stable periodic grid evolution and force for chemical cell masses.

    The diffusion/death update solves the semidiscrete grid equation exactly
    in Fourier space for a source held fixed over one time step.  This avoids
    the explicit five-point scheme's CFL restriction at the report's dt.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        n = config.grid_size
        h = 1.0 / n
        frequency = np.fft.fftfreq(n)
        eigenvalue = (4.0 / h**2) * (
            np.sin(np.pi * frequency[:, None]) ** 2
            + np.sin(np.pi * frequency[None, :]) ** 2
        )
        rate = 0.5 * config.alpha_chemical**2 * eigenvalue + config.death_rate
        self.decay = np.exp(-rate * config.dt)
        self.source_factor = np.empty_like(rate)
        np.divide(1.0 - self.decay, rate, out=self.source_factor, where=rate != 0)
        self.source_factor[rate == 0] = config.dt

        displacement = minimum_image(np.arange(n) / n)
        dx, dy = np.meshgrid(displacement, displacement, indexing="ij")
        nearby = (np.abs(dx) <= config.interaction_radius) & (
            np.abs(dy) <= config.interaction_radius
        )
        self.kernel_x_hat = np.fft.fft2(dx * nearby)
        self.kernel_y_hat = np.fft.fft2(dy * nearby)

    def advance(self, field: Array, source_positions: Array) -> Array:
        """Advance chemical cell mass one step with frozen particle sources."""
        n = self.config.grid_size
        source = deposit_cell_mass(
            source_positions,
            np.full(len(source_positions), self.config.mass * self.config.birth_rate),
            n,
        )
        return np.fft.ifft2(
            self.decay * np.fft.fft2(field) + self.source_factor * np.fft.fft2(source)
        ).real

    def force(self, field: Array, particles: Array) -> Array:
        """Convolve with the grid-sampled local kernel and interpolate.

        Bilinear interpolation smooths the hard cutoff within a grid cell, so
        this is an approximation to the continuous-position particle force.
        """
        field_hat = np.fft.fft2(field)
        fx = np.fft.ifft2(field_hat * self.kernel_x_hat).real
        fy = np.fft.ifft2(field_hat * self.kernel_y_hat).real
        return np.column_stack(
            (_sample_grid_center_values(fx, particles), _sample_grid_center_values(fy, particles))
        )


def advance_chemicals(
    chemicals: Array,
    source_positions: Array,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> Array:
    """Diffuse/kill existing chemicals and create Poisson births during a step.

    For particles frozen over the step, births get uniform event times, then
    independently diffuse and survive over their remaining lifetime.
    """
    dt = config.dt
    existing = (chemicals + config.alpha_chemical * np.sqrt(dt) * rng.standard_normal(chemicals.shape)) % 1.0
    surviving = existing[rng.random(len(existing)) < np.exp(-config.death_rate * dt)]

    births_per_particle = rng.poisson(config.birth_rate * dt, len(source_positions))
    if not np.any(births_per_particle):
        return surviving
    birth_sites = np.repeat(source_positions, births_per_particle, axis=0)
    remaining_time = dt * rng.random(len(birth_sites))
    born = (
        birth_sites
        + config.alpha_chemical
        * np.sqrt(remaining_time)[:, None]
        * rng.standard_normal(birth_sites.shape)
    ) % 1.0
    born = born[rng.random(len(born)) < np.exp(-config.death_rate * remaining_time)]
    return np.concatenate((surviving, born), axis=0)


@dataclass
class SimulationResult:
    config: SimulationConfig
    times: Array
    rms_distance: Array
    chemical_counts: NDArray[np.int64]
    field_total_mass: Array
    snapshot_times: Array
    stochastic_particles: Array
    field_particles: Array
    chemical_snapshots: list[Array]
    field_snapshots: Array

    def save(self, path: str | Path) -> None:
        """Save arrays without object/pickle fields for later thesis analysis."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        chemical_offsets = np.concatenate(
            ([0], np.cumsum([len(points) for points in self.chemical_snapshots]))
        )
        chemical_positions = np.concatenate(self.chemical_snapshots, axis=0)
        np.savez_compressed(
            path,
            config=json.dumps(asdict(self.config)),
            times=self.times,
            rms_distance=self.rms_distance,
            chemical_counts=self.chemical_counts,
            field_total_mass=self.field_total_mass,
            snapshot_times=self.snapshot_times,
            stochastic_particles=self.stochastic_particles,
            field_particles=self.field_particles,
            chemical_offsets=chemical_offsets,
            chemical_positions=chemical_positions,
            field_snapshots=self.field_snapshots,
        )


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run both coupled descriptions from the same initial state and Brownian noise."""
    init_seed, particle_seed, chemical_seed = np.random.SeedSequence(config.seed).spawn(3)
    initial_rng = np.random.default_rng(init_seed)
    particle_rng = np.random.default_rng(particle_seed)
    chemical_rng = np.random.default_rng(chemical_seed)

    initial_particles, chemicals = initial_positions(config, initial_rng)
    stochastic_particles = initial_particles.copy()
    field_particles = initial_particles.copy()
    field = deposit_cell_mass(
        chemicals, np.full(len(chemicals), config.mass), config.grid_size
    )
    field_solver = ChemicalField(config)

    times = np.arange(config.steps + 1, dtype=float) * config.dt
    rms = np.empty(config.steps + 1, dtype=float)
    counts = np.empty(config.steps + 1, dtype=np.int64)
    masses = np.empty(config.steps + 1, dtype=float)
    rms[0], counts[0], masses[0] = 0.0, len(chemicals), field.sum()

    snapshot_steps = list(range(0, config.steps + 1, config.snapshot_stride))
    if snapshot_steps[-1] != config.steps:
        snapshot_steps.append(config.steps)
    snapshot_index = set(snapshot_steps)
    stochastic_history = [stochastic_particles.copy()]
    field_history = [field_particles.copy()]
    chemical_history = [chemicals.copy()]
    field_history_mass = [field.copy()]

    for step in range(1, config.steps + 1):
        stochastic_drift = chemical_force(
            stochastic_particles, chemicals, config.mass, config.interaction_radius
        )
        field_drift = field_solver.force(field, field_particles)
        brownian = config.alpha_particle * np.sqrt(config.dt) * particle_rng.standard_normal(
            stochastic_particles.shape
        )
        next_stochastic_particles = (
            stochastic_particles + brownian + config.dt * stochastic_drift
        ) % 1.0
        next_field_particles = (field_particles + brownian + config.dt * field_drift) % 1.0
        chemicals = advance_chemicals(
            chemicals, stochastic_particles, config, chemical_rng
        )
        field = field_solver.advance(field, field_particles)
        stochastic_particles = next_stochastic_particles
        field_particles = next_field_particles

        rms[step] = torus_rms(stochastic_particles, field_particles)
        counts[step] = len(chemicals)
        masses[step] = field.sum()
        if step in snapshot_index:
            stochastic_history.append(stochastic_particles.copy())
            field_history.append(field_particles.copy())
            chemical_history.append(chemicals.copy())
            field_history_mass.append(field.copy())

    return SimulationResult(
        config=config,
        times=times,
        rms_distance=rms,
        chemical_counts=counts,
        field_total_mass=masses,
        snapshot_times=times[snapshot_steps],
        stochastic_particles=np.stack(stochastic_history),
        field_particles=np.stack(field_history),
        chemical_snapshots=chemical_history,
        field_snapshots=np.stack(field_history_mass),
    )
