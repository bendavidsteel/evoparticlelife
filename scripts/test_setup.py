#!/usr/bin/env python
"""Quick test to verify setup is working correctly."""

import os
# Configure JAX to not preallocate GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'  # Use only 25% of GPU memory

import jax
import jax.numpy as jnp
from evo_particle_life import ParticleLife, species_to_color
from metrics import MetricsTracker

print("Testing evolutionary particle life setup...")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Using device: {jax.devices()[0].device_kind}")

# Create small simulation
print("\nCreating small test simulation...")
sim = ParticleLife(
    num_particles=500,
    species_dims=2,
    size=jnp.array([2.0, 2.0]),
    n_dims=2,
    steps_per_frame=10,
)

# Initialize metrics
metrics_tracker = MetricsTracker(sim.displacement_fn)

# Run a few steps
print("Running 10 simulation steps...")
for i in range(10):
    if i == 0:
        # Time first step (includes JIT compilation)
        import time
        start = time.time()
        positions, step_count = sim.step_while(max_steps=50)
        end = time.time()
        print(f"  First step (with JIT): {end-start:.3f}s, {step_count} steps")
    else:
        positions, step_count = sim.step_while(max_steps=50)

# Compute metrics
print("\nComputing metrics...")
metrics = metrics_tracker.compute_all_metrics(sim, 10)

print("\nMetrics:")
print(f"  Kinetic energy: {metrics['kinetic_energy']:.4f}")
print(f"  Species entropy: {metrics['species_entropy']:.4f}")
print(f"  N clusters: {metrics['n_species_clusters']}")
print(f"  Mean velocity: {metrics['mean_velocity']:.4f}")
print(f"  Activity: {metrics['activity']:.4f}")

print("\n✓ Setup test completed successfully!")
print("\nNext steps:")
print("  1. Run a single experiment: python run_experiment.py")
print("  2. Use small config: python run_experiment.py simulation=small")
print("  3. Disable wandb: python run_experiment.py wandb.mode=disabled")
print("  4. Run sweep: python run_sweep.py experiment=sweep")
