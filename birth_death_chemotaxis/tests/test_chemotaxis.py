"""Numerical and stochastic checks for the rebuilt simulation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from chemotaxis import (
    ChemicalField,
    SimulationConfig,
    advance_chemicals,
    chemical_force,
    deposit_cell_mass,
    minimum_image,
    simulate,
    torus_rms,
    torus_wasserstein2,
)


class GeometryTests(unittest.TestCase):
    def test_wasserstein2_uses_periodic_optimal_matching(self) -> None:
        first = np.array([[0.99, 0.5], [0.25, 0.25]])
        second = np.array([[0.25, 0.25], [0.01, 0.5]])
        self.assertAlmostEqual(torus_wasserstein2(first, second), 0.02 / np.sqrt(2))
        self.assertLess(torus_wasserstein2(first, second), torus_rms(first, second))
        with self.assertRaises(ValueError):
            torus_wasserstein2(first, second[:1])

    def test_shortest_torus_distance_and_empty_force(self) -> None:
        point = np.array([[0.99, 0.5]])
        other = np.array([[0.01, 0.5]])
        np.testing.assert_allclose(minimum_image(point - other), [[-0.02, 0.0]])
        self.assertAlmostEqual(torus_rms(point, other), 0.02)
        np.testing.assert_array_equal(
            chemical_force(point, np.empty((0, 2)), 0.1, 0.25), np.zeros((1, 2))
        )

    def test_field_impulse_force_matches_particle_force_at_grid_centers(self) -> None:
        config = SimulationConfig(n_particles=1, steps=1, grid_size=32, chemical_mass=0.2)
        particle = np.array([[(5.5 / 32), (7.5 / 32)]])
        chemical = np.array([[(4.5 / 32), (7.5 / 32)]])
        field = deposit_cell_mass(chemical, np.array([config.mass]), config.grid_size)
        actual = ChemicalField(config).force(field, particle)
        expected = chemical_force(particle, chemical, config.mass, config.interaction_radius)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_grid_refinement_reduces_force_cutoff_bias(self) -> None:
        chemical = np.array([[0.005, 0.005]])
        particle = np.array([[0.26, 0.005]])  # just outside the radius 0.25 cutoff
        exact = chemical_force(particle, chemical, 1.0, 0.25)[0, 0]
        self.assertEqual(exact, 0.0)
        approximations = []
        for size in (100, 400):
            config = SimulationConfig(n_particles=1, steps=1, grid_size=size, chemical_mass=1.0)
            field = deposit_cell_mass(chemical, np.array([1.0]), size)
            approximations.append(ChemicalField(config).force(field, particle)[0, 0])
        self.assertGreater(abs(approximations[0]), 0.05)
        self.assertLess(abs(approximations[1]), 1e-10)


class ProcessTests(unittest.TestCase):
    def test_field_total_mass_obeys_birth_death_ode(self) -> None:
        config = SimulationConfig(
            n_particles=5, steps=40, dt=0.02, grid_size=24, snapshot_stride=10
        )
        positions = np.array(
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]]
        )
        field = deposit_cell_mass(
            positions, np.full(config.n_particles, config.mass), config.grid_size
        )
        solver = ChemicalField(config)
        for _ in range(config.steps):
            field = solver.advance(field, positions)
        t = config.steps * config.dt
        initial_mass = config.n_particles * config.mass
        stationary_mass = config.n_particles * config.mass * config.birth_rate / config.death_rate
        expected = stationary_mass + (initial_mass - stationary_mass) * np.exp(-config.death_rate * t)
        self.assertAlmostEqual(field.sum(), expected, places=11)
        self.assertGreaterEqual(field.min(), -1e-12)

    def test_stochastic_chemical_count_mean(self) -> None:
        config = SimulationConfig(
            n_particles=3,
            steps=20,
            dt=0.1,
            alpha_chemical=0,
            birth_rate=4,
            death_rate=2,
            chemical_mass=0.1,
        )
        rng = np.random.default_rng(287)
        positions = np.zeros((config.n_particles, 2))
        counts = []
        for _ in range(800):
            chemicals = np.empty((0, 2))
            for _ in range(config.steps):
                chemicals = advance_chemicals(chemicals, positions, config, rng)
            counts.append(len(chemicals))
        t = config.steps * config.dt
        expected = config.n_particles * config.birth_rate / config.death_rate * (
            1.0 - np.exp(-config.death_rate * t)
        )
        self.assertLess(abs(np.mean(counts) - expected), 0.35)

    def test_common_particle_noise_gives_identical_paths_without_chemicals(self) -> None:
        config = SimulationConfig(
            n_particles=5,
            steps=17,
            grid_size=20,
            birth_rate=0,
            death_rate=0,
            chemical_mass=0.1,
            initial_chemicals="none",
            snapshot_stride=4,
        )
        result = simulate(config)
        np.testing.assert_array_equal(result.stochastic_particles, result.field_particles)
        np.testing.assert_array_equal(result.rms_distance, np.zeros(config.steps + 1))
        np.testing.assert_array_equal(
            result.wasserstein2_distance, np.zeros(len(result.snapshot_times))
        )
        np.testing.assert_array_equal(result.chemical_counts, np.zeros(config.steps + 1))

    def test_seed_and_saved_arrays_are_reproducible(self) -> None:
        config = SimulationConfig(n_particles=7, steps=8, grid_size=16, seed=42)
        first = simulate(config)
        second = simulate(config)
        np.testing.assert_array_equal(first.stochastic_particles, second.stochastic_particles)
        np.testing.assert_array_equal(first.chemical_counts, second.chemical_counts)
        np.testing.assert_array_equal(first.wasserstein2_distance, second.wasserstein2_distance)
        snapshot_steps = np.rint(first.snapshot_times / config.dt).astype(int)
        self.assertTrue(
            np.all(first.wasserstein2_distance <= first.rms_distance[snapshot_steps] + 1e-12)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.npz"
            first.save(path)
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["rms_distance"], first.rms_distance)
                np.testing.assert_array_equal(
                    data["wasserstein2_distance"], first.wasserstein2_distance
                )
                offsets = data["chemical_offsets"]
                self.assertEqual(len(offsets), len(first.chemical_snapshots) + 1)
                self.assertEqual(offsets[-1], len(data["chemical_positions"]))


if __name__ == "__main__":
    unittest.main()
