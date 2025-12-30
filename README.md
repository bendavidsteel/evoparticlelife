# Evolutionary Particle Life

An evolutionary particle life simulation using JAX for high-performance computation. This system implements evolving particles with local interaction rules, species diversity through mutation and selection, and comprehensive experimental tracking.

## Features

### Core Simulation
- **JAX-accelerated physics**: Fast particle dynamics using JAX and jax-md
- **Periodic boundary conditions**: Toroidal space for continuous interaction
- **Neighbor lists**: Efficient spatial queries using jax-md neighbor lists
- **Species evolution**: Particles copy successful neighbors and mutate over time

### Circle-based Mutation System
- **Spatial mutations**: Random circular regions selected for mutation (not point mutations)
- **Three transformation types**:
  - **Noise addition**: Random gaussian noise to species vectors
  - **Rotation**: Species vectors rotated in 2D space
  - **Scaling**: Species vectors scaled by random factors
- **Group-level evolution**: Mutations affect spatially coherent groups

### Optimized JAX While Loop Stepping
- **Adaptive stepping**: Runs steps until neighbor list overflow
- **Maximized efficiency**: Reduces overhead by maximizing steps between rebuilds
- **Configurable limits**: Balance efficiency and memory usage

### Comprehensive Metrics System

#### Physics Metrics
- Momentum, mean velocity, velocity variance
- Spatial extent

#### Species Diversity Metrics
- Number of unique species
- Shannon entropy
- Species variance and range
- Pairwise diversity

#### Spatial Structure Metrics
- Activity level
- Clustering coefficient
- Nearest neighbor distances

#### Visual Complexity (Flow-Lenia inspired)
- **Compression complexity**: PNG compression ratio as proxy for Kolmogorov complexity
- **Spatial frequency**: FFT-based high-frequency content measure

Based on: [Flow-Lenia.png: Evolving Multi-Scale Complexity by Means of Compression](https://arxiv.org/abs/2408.06374)

### VLM Evaluation
- **Life-likeness scoring**: Uses Qwen3-VL-8B to evaluate "life-like" qualities
- **Local inference**: vLLM for fast, free evaluation
- **Temporal analysis**: Tracks growth, movement, interaction patterns

Inspired by: [ASAL: Agent-Supervised Artificial Life](https://arxiv.org/abs/2412.17799)

### Experiment Tracking
- **Wandb integration**: Track experiments, metrics, videos
- **Parameter sweeps**: Automated grid search over evolutionary parameters
- **Hydra configuration**: YAML-based config management

## Installation

```bash
# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# For CUDA support (GPU acceleration)
uv pip install --upgrade "jax[cuda12]"
```

## Quick Start

### Test Your Setup

```bash
python test_setup.py
```

### Run Single Experiment

```bash
# Default config
python run_experiment.py

# Small simulation for testing
python run_experiment.py simulation=small

# Disable wandb
python run_experiment.py wandb.mode=disabled

# Override parameters
python run_experiment.py \
    simulation.num_particles=2000 \
    mutation.mutation_prob=0.02
```

### Run Parameter Sweep

```bash
# Evolution-focused sweep
python run_sweep.py experiment=evolution_sweep

# Smaller focused sweep
python run_sweep.py experiment=evo_focused
```

### Analyze Sweep Results

```bash
python analyze_sweep.py
```

## Configuration

All experiments use Hydra YAML configs in `conf/`:

```
conf/
├── config.yaml              # Main config
├── simulation/
│   ├── default.yaml        # Default parameters
│   ├── small.yaml          # Small test
│   └── large.yaml          # Large simulation
├── mutation/
│   └── default.yaml        # Mutation parameters
├── metrics/
│   └── default.yaml        # Metrics config
└── experiment/
    ├── single.yaml         # Single run
    ├── evolution_sweep.yaml # Full sweep
    └── evo_focused.yaml    # Smaller sweep
```

Override from command line:

```bash
# Override parameters
python run_experiment.py simulation.mass=0.05 mutation.copy_prob=0.01

# Use different configs
python run_experiment.py simulation=large experiment=evolution_sweep

# See full config
python run_experiment.py --cfg job
```

## Key Parameters

### Evolutionary Parameters (main focus)
- `copy_dist`: Maximum distance for copying successful neighbors
- `copy_prob`: Probability of copying per frame
- `mutation_prob`: Probability of mutation per frame
- `species_dim`: Dimensionality of species vectors

### Physics Parameters
- `mass`: Particle mass
- `rmax`: Maximum interaction radius
- `repulsion_dist`: Repulsion distance
- `repulsion`: Repulsion strength

### Mutation Parameters
- `min_radius`: Minimum mutation circle radius
- `max_radius`: Maximum mutation circle radius

## Python API

```python
import jax.numpy as jnp
from evo_particle_life import ParticleLife
from metrics import MetricsTracker

# Create simulation
sim = ParticleLife(
    num_particles=4000,
    species_dims=2,
    size=jnp.array([3.0, 3.0]),
)

# Run with adaptive stepping
positions, step_count = sim.step_while(max_steps=200)

# Track metrics
tracker = MetricsTracker(sim.displacement_fn)
metrics = tracker.compute_all_metrics(sim, step=0)
```

## Research Questions

The metrics system enables investigation of:

1. **Species diversification dynamics**: How does diversity evolve over time?
2. **Complexity emergence**: When do complex visual patterns appear?
3. **VLM correlation**: Do compression-complex systems score higher on life-likeness?
4. **Parameter effects**: How do copy_prob, mutation_prob affect diversity/complexity?
5. **Phase transitions**: Identify sudden changes in system behavior

## Architecture

- **`evo_particle_life.py`**: Core simulation with JAX physics and mutation
- **`metrics.py`**: Comprehensive metrics (species, complexity, spatial)
- **`vlm_evaluator.py`**: VLM-based life-likeness evaluation
- **`run_experiment.py`**: Single experiment runner
- **`run_sweep.py`**: Parameter sweep runner
- **`analyze_sweep.py`**: Results analysis
- **`render.py`**: Fast JAX-based visualization

## Performance

- **JIT compilation**: All core functions JIT compiled
- **Neighbor lists**: O(N) force computation
- **While loop optimization**: Minimizes Python overhead
- **GPU compatible**: Runs on GPU with CUDA JAX

## References

Inspired by:
- [Flow-Lenia: Massively Parallel Continuous Cellular Automata](https://arxiv.org/abs/2212.07906)
- [Flow-Lenia.png: Evolving Multi-Scale Complexity by Means of Compression](https://arxiv.org/abs/2408.06374)
- [ASAL: Agent-Supervised Artificial Life](https://arxiv.org/abs/2412.17799)

## License

MIT
