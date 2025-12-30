"""Parameter sweep runner using Hydra multirun."""

import os
# Configure JAX GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import wandb
from pathlib import Path
from tqdm import tqdm
import numpy as np
import itertools

from evo_particle_life import ParticleLife, species_to_color
from metrics import MetricsTracker
from render import draw_particles_2d_fast


def run_single_config(cfg: DictConfig, run_id: int):
    """Run a single experiment configuration.

    Args:
        cfg: Hydra configuration
        run_id: Unique ID for this run
    """
    # Set random seed
    jax.config.update("jax_enable_x64", False)
    seed = cfg.seed + run_id if cfg.seed is not None else run_id
    key = jax.random.PRNGKey(seed)

    # Initialize wandb
    if cfg.wandb.mode != "disabled":
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            name=f"{cfg.experiment.name}_run{run_id}",
            reinit=True,
        )
    else:
        run = None

    # Create simulation
    size = jnp.array([cfg.simulation.box_size] * cfg.simulation.n_dims)

    sim = ParticleLife(
        num_particles=cfg.simulation.num_particles,
        species_dims=cfg.simulation.species_dim,
        size=size,
        n_dims=cfg.simulation.n_dims,
        dt=cfg.simulation.dt,
        steps_per_frame=cfg.simulation.steps_per_frame,
    )

    # Override default parameters with config
    sim.mass = cfg.simulation.mass
    sim.half_life = cfg.simulation.half_life
    sim.rmax = cfg.simulation.rmax
    sim.repulsion_dist = cfg.simulation.repulsion_dist
    sim.repulsion = cfg.simulation.repulsion

    # Override mutation parameters with config
    sim.mutation_prob = cfg.mutation.mutation_prob
    sim.max_copy_dist = cfg.mutation.max_copy_dist
    sim.max_species_dist = cfg.mutation.max_species_dist
    sim.copy_prob = cfg.mutation.copy_prob

    # Recompute damping coefficient with new parameters
    sim.damping = (0.5) ** (cfg.simulation.dt / cfg.simulation.half_life)

    # Update params namedtuple with all parameters
    sim.params = sim.params._replace(
        mass=cfg.simulation.mass,
        half_life=cfg.simulation.half_life,
        rmax=cfg.simulation.rmax,
        repulsion_dist=cfg.simulation.repulsion_dist,
        repulsion=cfg.simulation.repulsion,
        damping=sim.damping,
        mutation_prob=cfg.mutation.mutation_prob,
        max_copy_dist=cfg.mutation.max_copy_dist,
        max_species_dist=cfg.mutation.max_species_dist,
        copy_prob=cfg.mutation.copy_prob,
    )

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(sim.displacement_fn)

    # Run simulation
    step = 0
    num_steps = cfg.experiment.num_steps
    use_while_loop = cfg.simulation.use_while_loop
    max_steps_per_update = cfg.simulation.get('max_steps_per_update', 200)

    # Calculate logging interval in steps (not frames)
    log_interval_steps = cfg.output.log_frequency * cfg.simulation.steps_per_frame
    next_log_step = log_interval_steps

    # Image logging frequency (less frequent than metrics)
    image_log_interval = log_interval_steps * 5  # Every 5th metrics log
    next_image_step = image_log_interval

    # Track OOM status
    oom_occurred = False
    oom_step = None
    oom_error_msg = None

    try:
        while step < num_steps:
            # Step simulation
            if use_while_loop:
                positions, actual_steps = sim.step_while(max_steps=max_steps_per_update)
                step += int(actual_steps)
            else:
                positions = sim.step()
                step += cfg.simulation.steps_per_frame

            # Log metrics when we pass the next logging threshold
            if step >= next_log_step:
                metrics = metrics_tracker.compute_all_metrics(sim, step)

                # Add rendered image periodically
                if run is not None and step >= next_image_step:
                    try:
                        size = jnp.array([cfg.simulation.box_size] * cfg.simulation.n_dims)
                        colours = species_to_color(sim.species)
                        img_jax = draw_particles_2d_fast(sim.positions, colours, size, img_size=512)
                        # Convert JAX array to numpy for wandb
                        img_np = np.array(img_jax)
                        metrics['visualization'] = wandb.Image(img_np, caption=f"Step {step}")
                        next_image_step += image_log_interval
                    except Exception as e:
                        print(f"Warning: Image logging failed: {e}")

                if run is not None:
                    wandb.log(metrics, step=step)

                next_log_step += log_interval_steps

    except jax.errors.JaxRuntimeError as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
            oom_occurred = True
            oom_step = step
            oom_error_msg = str(e)
            print(f"\n⚠️  OOM at step {step}/{num_steps} ({100*step/num_steps:.1f}% complete)")
            print(f"   Saving metrics collected up to this point...")
        else:
            raise

    # Final metrics and image at end of run (or at OOM point)
    try:
        metrics = metrics_tracker.compute_all_metrics(sim, step)
    except Exception as e:
        print(f"Warning: Final metrics computation failed: {e}")
        metrics = {}
    if run is not None:
        try:
            size = jnp.array([cfg.simulation.box_size] * cfg.simulation.n_dims)
            colours = species_to_color(sim.species)
            img_jax = draw_particles_2d_fast(sim.positions, colours, size, img_size=512)
            # Convert JAX array to numpy for wandb
            img_np = np.array(img_jax)
            metrics['visualization'] = wandb.Image(img_np, caption=f"Final (Step {step})")
        except Exception as e:
            print(f"Warning: Final image logging failed: {e}")
        wandb.log(metrics, step=step)

    # Compute summary statistics
    summary = metrics_tracker.get_summary_statistics()

    # Add OOM metadata to summary
    summary['oom_occurred'] = oom_occurred
    if oom_occurred:
        summary['oom_step'] = oom_step
        summary['completion_ratio'] = step / num_steps
        summary['oom_error_message'] = oom_error_msg
        print(f"   Run completed with OOM (reached {100*step/num_steps:.1f}%)")
    else:
        summary['completion_ratio'] = 1.0
        print(f"✅ Run completed successfully")

    if run is not None:
        wandb.log(summary)
        wandb.finish()

    return summary


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run parameter sweep using configurations from sweep config.

    Use with Hydra's multirun feature or manual sweep over parameters.
    """
    print("Sweep Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Check if sweep parameters are defined
    if 'sweep' not in cfg and 'sweep' not in cfg.experiment:
        print("No sweep parameters defined. Running single experiment.")
        run_single_config(cfg, 0)
        return

    # Generate parameter combinations
    sweep_params = OmegaConf.to_container(cfg.experiment.sweep if 'sweep' in cfg.experiment else cfg.sweep, resolve=True)
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] if isinstance(sweep_params[name], list) else [sweep_params[name]]
                   for name in param_names]

    param_combinations = list(itertools.product(*param_values))

    print(f"\nRunning sweep over {len(param_combinations)} parameter combinations...")
    print(f"Parameters: {param_names}")

    results = []
    for i, params in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(param_combinations)}")
        print(f"{'='*60}")

        # Create modified config for this run
        cfg_run = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        # Update simulation parameters from sweep
        for name, value in zip(param_names, params):
            if name in ['mass', 'half_life', 'rmax', 'repulsion_dist', 'repulsion',
                       'num_particles', 'species_dim', 'box_size']:
                OmegaConf.update(cfg_run, f"simulation.{name}", value)
            elif name in ['mutation_prob', 'max_copy_dist', 'copy_prob', 'max_species_dist',
                         'min_radius', 'max_radius', 'noise_strength']:
                OmegaConf.update(cfg_run, f"mutation.{name}", value)
            print(f"  {name}: {value}")

        # Run experiment
        summary = run_single_config(cfg_run, i)
        results.append({
            'params': dict(zip(param_names, params)),
            'summary': summary
        })

        # Print key metrics
        print(f"\n  Final entropy: {summary.get('species_entropy_final', 0):.4f}")
        print(f"  Mean activity: {summary.get('activity_mean', 0):.4f}")

    print(f"\n{'='*60}")
    print("Sweep Complete!")
    print(f"Total experiments: {len(results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
