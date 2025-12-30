"""Test suite for validating simulation configurations before running long experiments.

This script tests whether a configuration can:
1. Initialize without errors
2. Run neighbor list allocation successfully
3. Execute multiple simulation steps without OOM
4. Compute all metrics without errors
"""

import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
from evo_particle_life import ParticleLife, species_to_color
from metrics import MetricsTracker
from render import draw_particles_2d_fast
import numpy as np


def test_initialization(cfg: DictConfig) -> bool:
    """Test if simulation can be initialized."""
    try:
        print("  ✓ Testing initialization...")
        size = jnp.array([cfg.simulation.box_size] * cfg.simulation.n_dims)

        sim = ParticleLife(
            num_particles=cfg.simulation.num_particles,
            species_dims=cfg.simulation.species_dim,
            size=size,
            n_dims=cfg.simulation.n_dims,
            dt=cfg.simulation.dt,
            steps_per_frame=cfg.simulation.steps_per_frame,
        )

        # Override default parameters with config (exactly as run_sweep.py does)
        sim.mass = cfg.simulation.mass
        sim.half_life = cfg.simulation.half_life
        sim.rmax = cfg.simulation.rmax
        sim.repulsion_dist = cfg.simulation.repulsion_dist
        sim.repulsion = cfg.simulation.repulsion

        sim.params = sim.params._replace(
            mass=cfg.simulation.mass,
            half_life=cfg.simulation.half_life,
            rmax=cfg.simulation.rmax,
            repulsion_dist=cfg.simulation.repulsion_dist,
            repulsion=cfg.simulation.repulsion,
        )

        print(f"    ✓ Initialized {cfg.simulation.num_particles} particles, species_dim={cfg.simulation.species_dim}")
        return True, sim
    except Exception as e:
        print(f"    ✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_neighbor_allocation(sim, cfg: DictConfig) -> bool:
    """Test if neighbor list can be allocated."""
    try:
        print("  ✓ Testing neighbor list allocation...")
        # Force neighbor list allocation
        if cfg.simulation.use_while_loop:
            sim.step_while(max_steps=cfg.simulation.get('max_steps_per_update', 200))
        else:
            sim.step()
        print("    ✓ Neighbor list allocated successfully")
        return True
    except Exception as e:
        print(f"    ✗ Neighbor allocation failed: {e}")
        return False


def test_multiple_steps(sim, cfg: DictConfig, num_steps: int = 10) -> bool:
    """Test if simulation can run multiple steps."""
    try:
        print(f"  ✓ Testing {num_steps} simulation steps...")
        for i in range(num_steps):
            if cfg.simulation.use_while_loop:
                sim.step_while(max_steps=cfg.simulation.get('max_steps_per_update', 200))
            else:
                sim.step()
        print(f"    ✓ Completed {num_steps} steps successfully")
        return True
    except Exception as e:
        print(f"    ✗ Step execution failed at step {i}: {e}")
        return False


def test_metrics_computation(sim, cfg: DictConfig) -> bool:
    """Test if all metrics can be computed."""
    try:
        print("  ✓ Testing metrics computation...")
        tracker = MetricsTracker(sim.displacement_fn)
        metrics = tracker.compute_all_metrics(sim, step=0)

        # Check all expected metrics are present
        expected_metrics = [
            'momentum', 'mean_velocity', 'velocity_variance', 'spatial_extent',
            'num_unique_species', 'species_entropy', 'species_variance',
            'species_range', 'species_pairwise_diversity', 'mean_nn_distance',
            'clustering_coeff', 'activity'
        ]

        missing = [m for m in expected_metrics if m not in metrics]
        if missing:
            print(f"    ⚠ Missing metrics: {missing}")

        print(f"    ✓ Computed {len(metrics)} metrics successfully")
        return True
    except Exception as e:
        print(f"    ✗ Metrics computation failed: {e}")
        return False


def test_rendering(sim, cfg: DictConfig) -> bool:
    """Test if visualization can be rendered."""
    try:
        print("  ✓ Testing rendering...")
        size = jnp.array([cfg.simulation.box_size] * cfg.simulation.n_dims)
        colours = species_to_color(sim.species)
        img_jax = draw_particles_2d_fast(sim.positions, colours, size, img_size=512)
        img_np = np.array(img_jax)

        if img_np.shape != (512, 512, 3):
            print(f"    ⚠ Unexpected image shape: {img_np.shape}")
            return False

        print(f"    ✓ Rendered {img_np.shape} image successfully")
        return True
    except Exception as e:
        print(f"    ✗ Rendering failed: {e}")
        return False


def test_long_run(sim, cfg: DictConfig, num_steps: int = 1000) -> bool:
    """Test if simulation can run for extended period (to catch delayed OOM)."""
    try:
        print(f"  ✓ Testing long run ({num_steps} steps)...")
        step_count = 0

        while step_count < num_steps:
            if cfg.simulation.use_while_loop:
                _, actual_steps = sim.step_while(max_steps=cfg.simulation.get('max_steps_per_update', 200))
                step_count += int(actual_steps)
            else:
                sim.step()
                step_count += cfg.simulation.steps_per_frame

            # Print progress every 200 steps
            if step_count % 200 == 0:
                print(f"    → Progress: {step_count}/{num_steps} steps")

        print(f"    ✓ Completed {step_count} steps successfully")
        return True
    except Exception as e:
        print(f"    ✗ Long run failed at step {step_count}: {e}")
        return False


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run all tests for a given configuration."""

    print("=" * 70)
    print("CONFIGURATION TEST SUITE")
    print("=" * 70)

    # Print configuration summary
    print("\nConfiguration:")
    print(f"  Particles: {cfg.simulation.num_particles}")
    print(f"  Species dim: {cfg.simulation.species_dim}")
    print(f"  Box size: {cfg.simulation.box_size}")
    print(f"  Capacity multiplier: {cfg.simulation.capacity_multiplier}")
    print(f"  Use while loop: {cfg.simulation.use_while_loop}")

    # Run tests
    tests = []

    # Test 1: Initialization
    print("\n[1/6] Initialization Test")
    success, sim = test_initialization(cfg)
    tests.append(("Initialization", success))
    if not success:
        print("\n❌ FAILED: Cannot proceed without successful initialization")
        return

    # Test 2: Neighbor allocation
    print("\n[2/6] Neighbor List Allocation Test")
    success = test_neighbor_allocation(sim, cfg)
    tests.append(("Neighbor Allocation", success))
    if not success:
        print("\n❌ FAILED: Cannot proceed without successful neighbor allocation")
        return

    # Test 3: Multiple steps
    print("\n[3/6] Multiple Steps Test")
    success = test_multiple_steps(sim, cfg, num_steps=10)
    tests.append(("Multiple Steps", success))
    if not success:
        print("\n⚠ WARNING: Basic stepping failed, skipping remaining tests")
        return

    # Test 4: Metrics
    print("\n[4/6] Metrics Computation Test")
    success = test_metrics_computation(sim, cfg)
    tests.append(("Metrics", success))

    # Test 5: Rendering
    print("\n[5/6] Rendering Test")
    success = test_rendering(sim, cfg)
    tests.append(("Rendering", success))

    # Test 6: Long run (most important - catches delayed OOM)
    print("\n[6/6] Long Run Test (1000 steps)")
    success = test_long_run(sim, cfg, num_steps=1000)
    tests.append(("Long Run", success))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, success in tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(success for _, success in tests)

    if all_passed:
        print("\n✅ ALL TESTS PASSED - Configuration is safe to use for long experiments")
    else:
        failed = [name for name, success in tests if not success]
        print(f"\n❌ TESTS FAILED: {', '.join(failed)}")
        print("   Do NOT use this configuration for long experiments!")

    print("=" * 70)


if __name__ == "__main__":
    main()
