"""Main experiment runner using Hydra configuration."""

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

from evo_particle_life import ParticleLife, species_to_color
from metrics import MetricsTracker
from render import draw_particles_2d_fast
from vlm_evaluator import VLMLifeEvaluator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Run a single experiment with Hydra configuration.

    Args:
        cfg: Hydra configuration
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    jax.config.update("jax_enable_x64", False)
    if cfg.seed is not None:
        key = jax.random.PRNGKey(cfg.seed)
    else:
        key = jax.random.PRNGKey(0)

    # Initialize wandb
    if cfg.wandb.mode != "disabled":
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            name=cfg.experiment.name,
        )
        print(f"Wandb run: {run.url}")
    else:
        print("Wandb disabled")

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

    # Initialize VLM evaluator if enabled
    vlm_evaluator = None
    vlm_enabled = cfg.experiment.get('vlm_evaluation', {}).get('enabled', False)
    if vlm_enabled:
        try:
            vlm_evaluator = VLMLifeEvaluator()
            print("VLM life-likeness evaluation enabled")
        except Exception as e:
            print(f"Warning: Could not initialize VLM evaluator: {e}")
            vlm_enabled = False

    # Setup output directory
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Video frames and trajectory storage
    frames = [] if cfg.output.save_video else None
    trajectory_snapshots = [] if (vlm_enabled and cfg.experiment.get('vlm_evaluation', {}).get('eval_trajectory', False)) else None

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

    print(f"\nRunning simulation for {num_steps} steps...")
    print(f"Using while loop: {use_while_loop}")
    print(f"Logging metrics every {log_interval_steps} steps")
    print(f"Logging images every {image_log_interval} steps")

    with tqdm(total=num_steps, desc="Simulating") as pbar:
        while step < num_steps:
            # Step simulation
            prev_step = step
            if use_while_loop:
                positions, actual_steps = sim.step_while(max_steps=max_steps_per_update)
                step += int(actual_steps)  # Convert JAX array to int
            else:
                positions = sim.step()
                step += cfg.simulation.steps_per_frame

            pbar.update(step - prev_step)

            # Log metrics when we pass the next logging threshold
            if step >= next_log_step:
                if cfg.metrics.compute.physics or cfg.metrics.compute.evolutionary or cfg.metrics.compute.behavioral:
                    metrics = metrics_tracker.compute_all_metrics(sim, step)

                    # Add rendered image periodically
                    if cfg.wandb.mode != "disabled" and step >= next_image_step:
                        try:
                            size = jnp.array([cfg.simulation.box_size] * cfg.simulation.n_dims)
                            colours = species_to_color(sim.species)
                            img = draw_particles_2d_fast(sim.positions, colours, size, img_size=512)
                            metrics['visualization'] = wandb.Image(img, caption=f"Step {step}")
                            next_image_step += image_log_interval
                        except Exception as e:
                            print(f"Warning: Image logging failed: {e}")

                    if cfg.wandb.mode != "disabled":
                        wandb.log(metrics, step=step)

                    # Print some metrics
                    print(f"\nStep {step}:")
                    print(f"  Momentum: {metrics.get('momentum', 0):.4f}")
                    print(f"  Species entropy: {metrics.get('species_entropy', 0):.4f}")
                    print(f"  Unique species: {metrics.get('num_unique_species', 0)}")

                next_log_step += log_interval_steps

            # VLM evaluation
            if vlm_enabled and vlm_evaluator is not None:
                eval_freq = cfg.experiment.get('vlm_evaluation', {}).get('eval_frequency', 1000)
                if step > 0 and step % eval_freq == 0:
                    print(f"\n  Evaluating life-likeness at step {step}...")
                    try:
                        box_size = sim.size[0] if hasattr(sim.size, '__len__') else sim.size
                        vlm_scores = vlm_evaluator.evaluate_life_likeness(
                            sim.positions, sim.species, box_size,
                            context=f"Step {step}/{num_steps}"
                        )
                        print(f"  Life-likeness: {vlm_scores.get('life_likeness', 0):.1f}/10")
                        if cfg.wandb.mode != "disabled":
                            wandb.log({f"vlm_{k}": v for k, v in vlm_scores.items()}, step=step)
                    except Exception as e:
                        print(f"  VLM evaluation failed: {e}")

            # Store trajectory snapshot for VLM temporal evaluation
            if trajectory_snapshots is not None and step % 500 == 0:
                trajectory_snapshots.append((sim.positions.copy(), sim.species.copy()))

            # Save frame if recording video
            if cfg.output.save_video and step % (cfg.simulation.steps_per_frame * 5) == 0:
                colours = species_to_color(sim.species)
                image = draw_particles_2d_fast(positions, colours, size, img_size=400)
                frames.append(np.array(image * 255).astype(np.uint8))

            update_amount = cfg.simulation.steps_per_frame if not use_while_loop else int(actual_steps)
            pbar.update(min(update_amount, num_steps - pbar.n))

    # Compute summary statistics
    summary = metrics_tracker.get_summary_statistics()
    print("\nSummary Statistics:")
    for key, value in summary.items():
        if 'mean' in key:
            print(f"  {key}: {value:.4f}")

    # VLM temporal evaluation if requested
    if vlm_enabled and trajectory_snapshots and vlm_evaluator is not None:
        print("\nEvaluating temporal dynamics...")
        try:
            box_size = sim.size[0] if hasattr(sim.size, '__len__') else sim.size
            temporal_scores = vlm_evaluator.evaluate_trajectory(
                trajectory_snapshots, box_size, num_snapshots=4
            )
            print("\nTemporal Life-likeness Scores:")
            for key, value in temporal_scores.items():
                print(f"  {key}: {value:.1f}/10")

            if cfg.wandb.mode != "disabled":
                wandb.log({f"vlm_temporal_{k}": v for k, v in temporal_scores.items()})
                summary.update({f"vlm_temporal_{k}_final": v for k, v in temporal_scores.items()})
        except Exception as e:
            print(f"Temporal VLM evaluation failed: {e}")

    if cfg.wandb.mode != "disabled":
        wandb.log(summary)

        # Save final frame as image
        if frames:
            wandb.log({"final_frame": wandb.Image(frames[-1])})

        # Create and log video
        if cfg.output.save_video and frames:
            video_path = output_dir / f"simulation_{run.id}.mp4"
            try:
                import imageio.v2 as iio
                w = iio.get_writer(
                    str(video_path),
                    format='FFMPEG',
                    mode='I',
                    fps=cfg.output.video_fps,
                    codec='libx264',
                    pixelformat='yuv420p'
                )
                for frame in frames:
                    w.append_data(frame)
                w.close()

                wandb.log({"simulation_video": wandb.Video(str(video_path))})
                print(f"\nVideo saved to: {video_path}")
            except Exception as e:
                print(f"Failed to save video: {e}")

        wandb.finish()

    print(f"\nExperiment complete!")


if __name__ == "__main__":
    main()
