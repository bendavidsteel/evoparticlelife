"""Metrics for evaluating evolutionary particle life simulations."""

import io

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy import ndimage


# =============================================================================
# Physics Metrics
# =============================================================================

@jax.jit
def compute_momentum(velocities, mass):
    """Compute total momentum magnitude."""
    total_momentum = mass * jnp.sum(velocities, axis=0)
    return jnp.linalg.norm(total_momentum)


@jax.jit
def compute_mean_velocity(velocities):
    """Compute mean speed of particles."""
    speeds = jnp.linalg.norm(velocities, axis=1)
    return jnp.mean(speeds)


@jax.jit
def compute_velocity_variance(velocities):
    """Compute variance in particle speeds."""
    speeds = jnp.linalg.norm(velocities, axis=1)
    return jnp.var(speeds)


@jax.jit
def compute_spatial_extent(positions, box_size):
    """Compute the spatial extent of particles (variance in positions)."""
    # Center positions
    centered = positions - jnp.mean(positions, axis=0)
    # Handle periodic boundaries
    centered = jnp.where(centered > box_size / 2, centered - box_size, centered)
    centered = jnp.where(centered < -box_size / 2, centered + box_size, centered)
    return jnp.mean(jnp.var(centered, axis=0))


# =============================================================================
# Species Diversity Metrics
# =============================================================================

@jax.jit
def compute_num_unique_species(species, threshold=0.1):
    """Count number of unique species based on distance threshold.

    Args:
        species: (N, species_dims) species vectors
        threshold: Distance threshold for considering species the same

    Returns:
        Approximate number of unique species
    """
    # Use fewer bins but int64 to avoid overflow for high dimensions
    # For 8D with 10 bins: 10^8 = 100M < 2^63, safe for int64
    n_bins = 10
    species_binned = jnp.floor((species + 1.0) * n_bins / 2.0).astype(jnp.int64)
    species_binned = jnp.clip(species_binned, 0, n_bins - 1)

    # Create unique identifier using int64 to prevent overflow
    species_ids = species_binned[:, 0].astype(jnp.int64)
    for i in range(1, species.shape[1]):
        species_ids = species_ids * n_bins + species_binned[:, i]

    # Count unique IDs - use a large enough size to capture diversity
    max_unique = min(species.shape[0], 10000)
    unique_ids = jnp.unique(species_ids, size=max_unique, fill_value=-1)
    # Count non-padding entries
    num_unique = jnp.sum(unique_ids >= 0)

    return num_unique


@jax.jit
def compute_species_diversity_entropy(species):
    """Compute species diversity using Shannon entropy on species vectors.

    Groups species into bins and computes entropy of the distribution.
    """
    # Discretize species vectors into bins
    n_bins = 10
    species_binned = jnp.floor(species * n_bins / 2 + n_bins / 2).astype(jnp.int32)
    species_binned = jnp.clip(species_binned, 0, n_bins - 1)

    # Create unique identifier for each species combination
    species_ids = species_binned[:, 0]
    for i in range(1, species.shape[1]):
        species_ids = species_ids * n_bins + species_binned[:, i]

    # Count occurrences
    unique_ids, counts = jnp.unique(species_ids, return_counts=True, size=species.shape[0])

    # Compute Shannon entropy
    probs = counts / jnp.sum(counts)
    # Filter out zero probabilities
    probs = jnp.where(probs > 0, probs, 1.0)  # Replace 0 with 1 to avoid log(0)
    entropy = -jnp.sum(jnp.where(counts > 0, probs * jnp.log(probs), 0.0))

    return entropy


@jax.jit
def compute_species_variance(species):
    """Compute variance across all species dimensions.

    Higher variance = more diverse species values.
    """
    return jnp.mean(jnp.var(species, axis=0))


@jax.jit
def compute_species_range(species):
    """Compute the range (max - min) across species dimensions.

    Higher range = species span more of the possible space.
    """
    ranges = jnp.max(species, axis=0) - jnp.min(species, axis=0)
    return jnp.mean(ranges)


def compute_species_pairwise_diversity(species, key=None):
    """Compute mean pairwise distance between species.

    Higher value = species are more different from each other.

    Args:
        species: (N, species_dims) species vectors
        key: Optional JAX random key. If None, uses step-based deterministic sampling.
    """
    # Sample random pairs to avoid O(N^2) computation
    N = species.shape[0]
    n_samples = min(1000, N)

    if key is None:
        # Use deterministic but varied sampling based on species statistics
        # This avoids the fixed seed problem while remaining JIT-compatible
        seed = int(jnp.abs(jnp.sum(species[:10, 0]) * 1e6)) % (2**31)
        key = jax.random.PRNGKey(seed)

    # Generate random indices
    key1, key2 = jax.random.split(key)
    idx1 = jax.random.randint(key1, (n_samples,), 0, N)
    idx2 = jax.random.randint(key2, (n_samples,), 0, N)

    # Compute pairwise distances
    dists = jnp.linalg.norm(species[idx1] - species[idx2], axis=1)
    return jnp.mean(dists)


# =============================================================================
# Spatial Structure Metrics
# =============================================================================

def compute_nearest_neighbor_distances(positions, neighbors, k=5):
    """Compute mean distance to k nearest neighbors using neighbor list.

    Uses the existing spatial partitioning from neighbor list to avoid O(N²).
    """
    # Get neighbor indices (shape: [N, max_neighbors])
    neighbor_idx = neighbors.idx

    # Compute distances to each neighbor
    def compute_neighbor_dists(i):
        pos_i = positions[i]
        # Get valid neighbors (neighbor_idx uses -1 for empty slots)
        neighs = neighbor_idx[i]
        valid_mask = neighs >= 0

        # Compute distances to valid neighbors
        def dist_to_j(j_idx):
            j = neighs[j_idx]
            # Safe indexing - if j < 0, distance will be masked out
            pos_j = jnp.where(j >= 0, positions[jnp.maximum(j, 0)], pos_i)
            return jnp.linalg.norm(pos_j - pos_i)

        dists = jax.vmap(dist_to_j)(jnp.arange(len(neighs)))
        # Mask invalid neighbors with large distance
        dists = jnp.where(valid_mask, dists, 1e10)
        return dists

    all_neighbor_dists = jax.vmap(compute_neighbor_dists)(jnp.arange(len(positions)))

    # Sort and take mean of k smallest
    sorted_dists = jnp.sort(all_neighbor_dists, axis=1)
    # Skip first entry (self or zero distance) and take next k
    mean_k_nearest = jnp.mean(sorted_dists[:, :k])

    return mean_k_nearest


def compute_clustering_coefficient(positions, neighbors, radius=0.1):
    """Compute spatial clustering coefficient using neighbor list.

    Measures how clustered particles are in space.
    Uses existing spatial partitioning to avoid O(N²).
    """
    # Get neighbor indices
    neighbor_idx = neighbors.idx

    # Count valid neighbors for each particle
    def count_neighbors(i):
        neighs = neighbor_idx[i]
        valid_mask = neighs >= 0
        return jnp.sum(valid_mask)

    neighbor_counts = jax.vmap(count_neighbors)(jnp.arange(len(positions)))
    mean_neighbors = jnp.mean(neighbor_counts)

    # Normalize by typical neighbor count
    N = positions.shape[0]
    clustering = mean_neighbors / jnp.maximum(1.0, float(N) * 0.1)  # Normalize by 10% of particles

    return clustering


# =============================================================================
# Image Complexity Metrics (Flow-Lenia inspired)
# =============================================================================

# Lazy import cache to avoid repeated imports in hot path
_render_imports = {}


def _get_render_functions():
    """Lazy load render functions to avoid import overhead in hot path."""
    if not _render_imports:
        from evo_particle_life import species_to_color
        from render import draw_particles_2d_fast
        _render_imports['species_to_color'] = species_to_color
        _render_imports['draw_particles_2d_fast'] = draw_particles_2d_fast
    return _render_imports['species_to_color'], _render_imports['draw_particles_2d_fast']


def _render_to_pil(positions, species, box_size, img_size):
    """Render simulation state to PIL Image."""
    species_to_color, draw_particles_2d_fast = _get_render_functions()

    size = jnp.array([box_size, box_size]) if not hasattr(box_size, '__len__') else box_size
    colours = species_to_color(species)
    image = draw_particles_2d_fast(positions, colours, size, img_size=img_size)

    # Convert to PIL Image
    img_array = np.array(image)
    return Image.fromarray((img_array * 255).astype(np.uint8))


def _get_compression_ratio(image):
    """Get PNG compression ratio for an image."""
    img_array = np.array(image)
    uncompressed_size = img_array.nbytes

    buffer = io.BytesIO()
    image.save(buffer, format='PNG', compress_level=9)
    compressed_size = buffer.tell()

    return compressed_size / uncompressed_size


def compute_multiscale_complexity(positions, species, box_size, base_size=256, num_scales=4, target=None):
    """Compute multi-scale complexity following Flow-Lenia methodology.

    Based on "Flow-Lenia.png: Evolving Multi-Scale Complexity by Means of Compression"
    (arXiv:2408.06374).

    The key insight is that true complexity exists between order (highly compressible)
    and chaos (incompressible noise). We measure complexity at multiple scales and
    can optionally target a specific complexity level.

    Args:
        positions: Particle positions
        species: Species vectors
        box_size: Simulation box size
        base_size: Base resolution for rendering (default 256)
        num_scales: Number of scales to evaluate (default 4: 256, 128, 64, 32)
        target: Optional target complexity in [0, 1]. If provided, returns
                deviation from target (lower = closer to target complexity).

    Returns:
        Dictionary with complexity metrics at each scale and aggregate measures.
    """
    # Render at base resolution
    image = _render_to_pil(positions, species, box_size, base_size)

    complexities = []

    for scale_idx in range(num_scales):
        # Downsample by factor of 2^scale_idx
        scale_factor = 2 ** scale_idx
        current_size = base_size // scale_factor

        if current_size < 16:  # Minimum meaningful size
            break

        # Resize image to current scale
        scaled_image = image.resize((current_size, current_size), Image.Resampling.LANCZOS)

        # Compute compression ratio at this scale
        ratio = _get_compression_ratio(scaled_image)

        # Convert to complexity measure
        # Key insight from Flow-Lenia: we want the INVERSE of compressibility
        # But pure noise (incompressible) isn't complex, it's random
        # True complexity is INTERMEDIATE compressibility

        # Normalize: uniform → ~0.05, noise → ~0.95, patterns → 0.3-0.7
        complexities.append(ratio)

    # Compute aggregate complexity (mean across scales)
    mean_complexity = np.mean(complexities)

    # Compute "interesting complexity" - how far from extremes (order/chaos)
    # Peak interestingness at compression ratio ~0.4-0.6
    # This captures the "edge of chaos" where complex patterns emerge
    interesting_complexity = 1.0 - abs(mean_complexity - 0.5) * 2
    interesting_complexity = float(np.clip(interesting_complexity, 0.0, 1.0))

    # If target specified, compute deviation
    target_deviation = None
    if target is not None:
        # Fitness is how close we are to target (averaged across scales)
        target_deviation = float(np.mean([abs(c - target) for c in complexities]))

    result = {
        'multiscale_complexity': float(mean_complexity),
        'interesting_complexity': interesting_complexity,
        'complexity_by_scale': {f'scale_{2**i}': float(c) for i, c in enumerate(complexities)},
    }

    if target_deviation is not None:
        result['target_deviation'] = target_deviation

    return result


def compute_image_complexity(positions, species, box_size, img_size=256):
    """Compute visual complexity metrics (backward compatible interface).

    This is a simplified interface that calls the more comprehensive
    compute_multiscale_complexity internally.

    Args:
        positions: Particle positions
        species: Species vectors
        box_size: Simulation box size
        img_size: Resolution for rendering

    Returns:
        Dictionary with compression_complexity, normalized_complexity, spatial_frequency
    """
    # Get multi-scale metrics
    ms_result = compute_multiscale_complexity(positions, species, box_size, base_size=img_size)

    # Render for spatial frequency analysis
    image = _render_to_pil(positions, species, box_size, img_size)
    img_array = np.array(image)

    # Compute spatial frequency content
    gray = np.mean(img_array, axis=2) / 255.0
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    power_spectrum = np.abs(fft_shift)**2

    center_size = img_size // 4
    h, w = power_spectrum.shape
    center = power_spectrum[
        h//2 - center_size:h//2 + center_size,
        w//2 - center_size:w//2 + center_size
    ]
    center_power = np.sum(center)
    total_power = np.sum(power_spectrum) + 1e-10
    spatial_frequency = float(1.0 - center_power / total_power)

    # Use "interesting complexity" as the main normalized metric
    # This is 1.0 at the edge of chaos, 0.0 at pure order or pure noise
    return {
        'compression_complexity': ms_result['multiscale_complexity'],
        'normalized_complexity': ms_result['interesting_complexity'],
        'spatial_frequency': spatial_frequency,
    }


def compute_open_endedness(current_embedding, history_embeddings, method='nearest_neighbor'):
    """Compute open-endedness score based on ASAL methodology.

    Based on "Automating the Search for Artificial Life with Foundation Models"
    (arXiv:2412.17799).

    Measures how novel the current state is compared to all previous states.
    Higher scores indicate the simulation is exploring new territory.

    Args:
        current_embedding: Feature vector for current state (e.g., from compression or VLM)
        history_embeddings: List of previous embeddings
        method: 'nearest_neighbor' (ASAL style) or 'mean_distance'

    Returns:
        Open-endedness score (higher = more novel)
    """
    if len(history_embeddings) == 0:
        return 1.0  # First state is maximally novel

    current = np.array(current_embedding)
    history = np.array(history_embeddings)

    if method == 'nearest_neighbor':
        # ASAL style: novelty = 1 - max similarity to any previous state
        # Using cosine similarity
        current_norm = current / (np.linalg.norm(current) + 1e-10)
        history_norm = history / (np.linalg.norm(history, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(history_norm, current_norm)
        max_similarity = np.max(similarities)
        return float(1.0 - max_similarity)
    else:
        # Mean distance to all previous states
        distances = np.linalg.norm(history - current, axis=1)
        return float(np.mean(distances))


def compute_state_embedding(positions, species, box_size, img_size=64):
    """Compute a feature embedding for the current simulation state.

    This creates a compact representation that can be used for:
    - Open-endedness calculation
    - Diversity measurement
    - Similarity comparison

    Uses a combination of image statistics and species distribution.

    Args:
        positions: Particle positions
        species: Species vectors
        box_size: Simulation box size
        img_size: Resolution for image-based features

    Returns:
        1D numpy array embedding
    """
    # Image-based features
    image = _render_to_pil(positions, species, box_size, img_size)
    img_array = np.array(image).astype(np.float32) / 255.0

    # Color histogram (8 bins per channel)
    hist_r = np.histogram(img_array[:, :, 0], bins=8, range=(0, 1))[0]
    hist_g = np.histogram(img_array[:, :, 1], bins=8, range=(0, 1))[0]
    hist_b = np.histogram(img_array[:, :, 2], bins=8, range=(0, 1))[0]
    color_features = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
    color_features = color_features / (color_features.sum() + 1e-10)

    # Spatial statistics (mean, std of each channel in 4x4 grid)
    grid_size = 4
    cell_h, cell_w = img_size // grid_size, img_size // grid_size
    spatial_features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = img_array[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            spatial_features.extend([cell.mean(), cell.std()])
    spatial_features = np.array(spatial_features, dtype=np.float32)

    # Species distribution features
    species_np = np.array(species)
    species_mean = np.mean(species_np, axis=0)
    species_std = np.std(species_np, axis=0)
    species_features = np.concatenate([species_mean, species_std]).astype(np.float32)

    # Compression ratio as complexity feature
    ratio = _get_compression_ratio(image)
    complexity_features = np.array([ratio], dtype=np.float32)

    # Combine all features
    embedding = np.concatenate([
        color_features,
        spatial_features,
        species_features,
        complexity_features
    ])

    return embedding


@jax.jit
def compute_activity_level(velocities, positions, prev_positions, dt, steps_elapsed=1):
    """Compute activity level based on motion and velocity changes.

    Args:
        velocities: Current particle velocities
        positions: Current particle positions
        prev_positions: Positions at last measurement
        dt: Time step size
        steps_elapsed: Number of simulation steps since prev_positions was recorded

    Returns:
        Activity level (displacement rate + mean speed)
    """
    # Mean displacement since last measurement
    displacement = jnp.mean(jnp.linalg.norm(positions - prev_positions, axis=1))

    # Mean speed (instantaneous)
    mean_speed = jnp.mean(jnp.linalg.norm(velocities, axis=1))

    # Normalize displacement by actual elapsed time (steps_elapsed * dt)
    elapsed_time = steps_elapsed * dt
    displacement_rate = displacement / jnp.maximum(elapsed_time, 1e-10)

    # Combined activity metric
    activity = displacement_rate + mean_speed

    return activity


# =============================================================================
# Metrics Tracker
# =============================================================================

class MetricsTracker:
    """Tracks metrics over time for a particle life simulation."""

    def __init__(self, displacement_fn):
        self.displacement_fn = displacement_fn
        self.history = {
            'step': [],
            'momentum': [],
            'mean_velocity': [],
            'velocity_variance': [],
            'spatial_extent': [],
            'num_unique_species': [],
            'species_entropy': [],
            'species_variance': [],
            'species_range': [],
            'species_pairwise_diversity': [],
            'mean_nn_distance': [],
            'clustering_coeff': [],
            'activity': [],
            'compression_complexity': [],
            'normalized_complexity': [],
            'spatial_frequency': [],
            # New Flow-Lenia inspired metrics
            'multiscale_complexity': [],
            'interesting_complexity': [],
            'open_endedness': [],
        }
        self.prev_positions = None
        self.prev_step = 0  # Track step count for proper activity normalization
        self.prev_frame_embeddings = []  # For open-endedness calculation

    def compute_all_metrics(self, sim, step):
        """Compute all metrics for current simulation state."""
        metrics = {}

        # Basic physics metrics
        metrics['momentum'] = float(compute_momentum(sim.velocities, sim.mass))
        metrics['mean_velocity'] = float(compute_mean_velocity(sim.velocities))
        metrics['velocity_variance'] = float(compute_velocity_variance(sim.velocities))

        # Spatial metrics
        box_size = sim.size[0] if hasattr(sim.size, '__len__') else sim.size
        metrics['spatial_extent'] = float(compute_spatial_extent(sim.positions, box_size))

        # Species diversity metrics
        metrics['num_unique_species'] = int(compute_num_unique_species(sim.species))
        metrics['species_entropy'] = float(compute_species_diversity_entropy(sim.species))
        metrics['species_variance'] = float(compute_species_variance(sim.species))
        metrics['species_range'] = float(compute_species_range(sim.species))
        metrics['species_pairwise_diversity'] = float(compute_species_pairwise_diversity(sim.species))

        # Spatial structure metrics
        metrics['mean_nn_distance'] = float(compute_nearest_neighbor_distances(
            sim.positions, sim.neighbors, k=5
        ))
        metrics['clustering_coeff'] = float(compute_clustering_coefficient(
            sim.positions, sim.neighbors, radius=sim.rmax
        ))

        # Activity metrics - properly normalized by elapsed steps
        if self.prev_positions is not None:
            steps_elapsed = max(1, step - self.prev_step)
            metrics['activity'] = float(compute_activity_level(
                sim.velocities, sim.positions, self.prev_positions, sim.dt, steps_elapsed
            ))
        else:
            metrics['activity'] = 0.0

        self.prev_positions = sim.positions.copy()
        self.prev_step = step

        # Image complexity metrics (computed less frequently as they're slow)
        if step % 500 == 0:
            try:
                # Use 256x256 for better pattern detection (Flow-Lenia paper recommendation)
                complexity_metrics = compute_image_complexity(
                    sim.positions, sim.species, box_size, img_size=256
                )
                metrics.update(complexity_metrics)

                # Compute multi-scale complexity with more detail
                ms_metrics = compute_multiscale_complexity(
                    sim.positions, sim.species, box_size, base_size=256, num_scales=4
                )
                metrics['multiscale_complexity'] = ms_metrics['multiscale_complexity']
                metrics['interesting_complexity'] = ms_metrics['interesting_complexity']

                # Compute state embedding and open-endedness
                embedding = compute_state_embedding(sim.positions, sim.species, box_size)
                oe_score = compute_open_endedness(embedding, self.prev_frame_embeddings)
                metrics['open_endedness'] = oe_score

                # Store embedding for future comparison (keep last 100)
                self.prev_frame_embeddings.append(embedding)
                if len(self.prev_frame_embeddings) > 100:
                    self.prev_frame_embeddings = self.prev_frame_embeddings[-100:]

            except Exception as e:
                print(f"Warning: Image complexity computation failed: {e}")
                metrics['compression_complexity'] = 0.0
                metrics['spatial_frequency'] = 0.0
                metrics['multiscale_complexity'] = 0.0
                metrics['open_endedness'] = 0.0

        # Store in history
        metrics['step'] = step
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        return metrics

    def get_summary_statistics(self):
        """Compute summary statistics over the entire run."""
        summary = {}

        for key in self.history.keys():
            if key == 'step':
                continue

            values = np.array(self.history[key])
            if len(values) > 0:
                summary[f'{key}_mean'] = float(np.mean(values))
                summary[f'{key}_std'] = float(np.std(values))
                summary[f'{key}_min'] = float(np.min(values))
                summary[f'{key}_max'] = float(np.max(values))
                summary[f'{key}_final'] = float(values[-1])

        return summary
