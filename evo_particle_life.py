import collections
import functools

import jax
import jax.numpy as jnp
from jax_md import space, partition
import numpy as np
import pygame
from tqdm import tqdm

from render import draw_particles_2d_fast, sonify_particles

Params = collections.namedtuple('Params', [
    'mass',
    'half_life',
    'dt',
    'rmax',
    'repulsion_dist',
    'repulsion',
    # Precomputed values
    'damping',  # Precomputed (0.5) ** (dt / half_life)
    # Mutation parameters
    'mutation_prob',
    'max_copy_dist',
    'max_species_dist',
    'copy_prob',
])

@jax.jit
def force_graph(r, rmax, alpha, repulsion_dist, repulsion):
    """Compute force magnitude based on distance.

    Uses jnp.select for cleaner conditional logic.
    """
    repulsion_force = jnp.maximum(repulsion_dist - r, 0.) * -repulsion
    attraction_force = alpha * jnp.maximum(1 - abs(2 * r - rmax - repulsion_dist) / (rmax - repulsion_dist), 0)

    return jnp.select(
        [r < repulsion_dist, r < rmax],
        [repulsion_force, attraction_force],
        default=0.0
    )


@functools.partial(jax.jit, static_argnames=['displacement_fn'])
def compute_forces_with_neighbors(positions, species, alpha, neighbor_idx, rmax, repulsion_dist, repulsion, displacement_fn):
    """Compute forces using jax-md neighbor lists.

    Optimized version using masking instead of per-neighbor branching.

    Args:
        positions: (N, 2) particle positions
        species: (N, species_dims) species vectors
        alpha: (N, species_dims) interaction matrix
        neighbor_idx: (N, max_neighbors) neighbor indices from jax-md
        rmax: interaction radius
        repulsion_dist: repulsion distance
        repulsion: repulsion strength
        displacement_fn: displacement function from jax-md

    Returns:
        forces: (N, 2) force vectors
    """
    def compute_particle_force(pos_i, alpha_i, neighbors):
        """Compute force on particle i from its neighbors using masking."""
        # Get neighbor positions and species (use index 0 for invalid neighbors, will be masked)
        safe_neighbors = jnp.maximum(neighbors, 0)
        n_positions = positions[safe_neighbors]
        n_species = species[safe_neighbors]

        # Compute displacements vectorized (no per-neighbor function calls)
        dr = jax.vmap(lambda n_x: displacement_fn(n_x, pos_i))(n_positions)
        r = jnp.sqrt(jnp.sum(jnp.square(dr), axis=1)).clip(1e-10)

        # Compute interactions vectorized
        interactions = jnp.dot(n_species, alpha_i)

        # Compute force scalars vectorized
        force_scalars = jax.vmap(
            lambda r_, a_: force_graph(r_, rmax, a_, repulsion_dist, repulsion)
        )(r, interactions)

        # Compute force vectors
        directions = dr / (r[:, jnp.newaxis] + 1e-10)
        forces = directions * force_scalars[:, jnp.newaxis]

        # Mask invalid neighbors (no branching, just zero out)
        valid_mask = (neighbors >= 0)[:, jnp.newaxis]
        forces = forces * valid_mask

        return forces.sum(axis=0)

    # Vectorize over all particles
    forces = jax.vmap(compute_particle_force)(positions, alpha, neighbor_idx)
    return forces


@functools.partial(jax.jit, static_argnames=['displacement_fn', 'shift_fn'])
def compute_step(x, v, species, alpha, neighbor_idx, mass, damping, dt, rmax,
                 repulsion_dist, repulsion, displacement_fn, shift_fn):
    """Compute simulation step using jax-md neighbor lists.

    Args:
        x: (N, 2) positions
        v: (N, 2) velocities
        species: (N, species_dims) species vectors
        alpha: (N, species_dims) interaction matrix
        neighbor_idx: (N, max_neighbors) neighbor indices
        mass: scalar mass
        damping: precomputed damping coefficient (0.5) ** (dt / half_life)
        dt: time step
        rmax: interaction radius
        repulsion_dist: repulsion distance
        repulsion: repulsion strength
        displacement_fn: displacement function from jax-md

    Returns:
        x: updated positions
        v: updated velocities
    """
    # Compute forces
    f = compute_forces_with_neighbors(
        x, species, alpha, neighbor_idx, rmax, repulsion_dist, repulsion, displacement_fn
    )

    # Update velocities and positions (using precomputed damping)
    acc = f / mass
    v = damping * v + acc * dt
    x = shift_fn(x, v * dt)

    return x, v

@functools.partial(jax.jit, static_argnames=['displacement_fn'])
def copy_species_with_neighbors(subkeys, species, alpha, positions, neighbor_idx,
                                max_copy_dist=0.08, max_species_dist=0.2, copy_prob=0.001,
                                displacement_fn=None):
    """Copy species using jax-md neighbor lists.

    Optimized version using masking instead of per-neighbor branching.

    Args:
        subkeys: Random keys
        species: (N, species_dims) species vectors
        alpha: (N, species_dims) interaction matrix
        positions: (N, 2) particle positions
        neighbor_idx: (N, max_neighbors) neighbor indices
        max_copy_dist: Maximum distance for copying
        max_species_dist: Maximum species difference for copying
        copy_prob: Probability of copying per particle
        displacement_fn: displacement function from jax-md

    Returns:
        species: Updated species vectors
        alpha: Updated interaction matrix
    """
    N = species.shape[0]
    LOG10 = jnp.float32(2.302585)  # Precomputed ln(10)

    def compute_copy_probs(i, pos_i, species_i, neighbors):
        """Compute copy probabilities using masking instead of branching."""
        # Safe indexing for invalid neighbors
        safe_neighbors = jnp.maximum(neighbors, 0)
        n_positions = positions[safe_neighbors]
        n_species = species[safe_neighbors]

        # Compute distances vectorized
        dr = jax.vmap(lambda n_x: displacement_fn(n_x, pos_i))(n_positions)
        r = jnp.sqrt(jnp.sum(dr ** 2, axis=1)).clip(1e-10)

        # Compute species differences vectorized
        species_diff = jnp.linalg.norm(n_species - species_i, axis=1)

        # Check conditions (vectorized)
        within_dist = r < max_copy_dist
        within_species = species_diff < max_species_dist
        valid_neighbor = neighbors >= 0
        not_self = neighbors != i
        valid = within_dist & within_species & valid_neighbor & not_self

        # Compute probability factors
        dist_factor = 1.0 - r / max_copy_dist
        # Optimized: exp(-species_diff * ln(10)) instead of pow(10, -species_diff)
        species_factor = jnp.exp(-species_diff * LOG10)

        # Apply mask (no branching)
        probs = jnp.where(valid, dist_factor * species_factor, 0.0)

        return probs, neighbors

    # Get copy probabilities for all particles
    all_probs, all_indices = jax.vmap(compute_copy_probs)(
        jnp.arange(N), positions, species, neighbor_idx
    )

    # Normalize probabilities
    row_sums = all_probs.sum(axis=1, keepdims=True)
    all_probs = jnp.where(row_sums > 0, all_probs / row_sums, 0.0)

    # Sample which neighbor to copy from
    copy_sources_local = jax.random.categorical(subkeys[0], jnp.log(all_probs + 1e-10), axis=1)

    # Get actual particle indices
    copy_sources = all_indices[jnp.arange(N), copy_sources_local]

    # Sample which particles will copy
    copy_mask = jax.random.uniform(subkeys[1], (N,)) < copy_prob
    has_valid_neighbor = (row_sums[:, 0] > 0)
    copy_mask = copy_mask & has_valid_neighbor

    # Perform copying
    species = jnp.where(copy_mask[:, jnp.newaxis], species[copy_sources], species)
    alpha = jnp.where(copy_mask[:, jnp.newaxis], alpha[copy_sources], alpha)

    return species, alpha

@functools.partial(jax.jit, static_argnames=['displacement_fn', 'shift_fn'])
def multi_step(carry, _, species=None, alpha=None, params=None, displacement_fn=None, shift_fn=None):
    x, v, neighbors = carry

    # Update neighbor list
    neighbors = neighbors.update(x)

    # Compute step (using precomputed damping)
    x, v = compute_step(
        x, v, species, alpha, neighbors.idx,
        params.mass, params.damping, params.dt,
        params.rmax, params.repulsion_dist, params.repulsion,
        displacement_fn, shift_fn
    )

    return (x, v, neighbors), x

@functools.partial(jax.jit, static_argnames=['displacement_fn'])
def mutate_circle(key, positions, species, alpha, box_size,
                  mutation_prob=0.01, min_radius=0.05, max_radius=0.3,
                  displacement_fn=None):
    """Apply random transformation to particles within a randomly chosen circle.

    Optimized version using jax.lax.switch for transformation selection.

    Args:
        key: Random key
        positions: (N, 2) particle positions
        species: (N, species_dims) species vectors
        alpha: (N, species_dims) interaction matrix
        box_size: Size of simulation box
        mutation_prob: Probability of mutation occurring
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        displacement_fn: Optional displacement function for periodic boundaries

    Returns:
        species: Updated species vectors
        alpha: Updated interaction matrix
    """
    key, *subkeys = jax.random.split(key, 7)
    n_dims = positions.shape[1]
    n_species_dims = species.shape[1]

    # Decide whether to mutate
    do_mutate = jax.random.uniform(subkeys[0]) < mutation_prob

    # Sample random circle center
    center = jax.random.uniform(subkeys[1], (n_dims,), minval=0.0, maxval=box_size)

    # Sample random radius
    radius = jax.random.uniform(subkeys[2], minval=min_radius, maxval=max_radius)

    # Compute distances from center using displacement_fn if available
    if displacement_fn is not None:
        diffs = jax.vmap(lambda p: displacement_fn(p, center))(positions)
    else:
        # Fallback: manual periodic boundary handling
        diffs = positions - center
        diffs = jnp.where(diffs > box_size / 2, diffs - box_size, diffs)
        diffs = jnp.where(diffs < -box_size / 2, diffs + box_size, diffs)
    distances = jnp.linalg.norm(diffs, axis=1)

    # Create mask for particles within circle
    in_circle = distances < radius

    # Sample transformation type: 0=add noise, 1=rotate, 2=scale
    transform_type = jax.random.randint(subkeys[3], (), 0, 3)

    # Define transformation functions (only one will be executed)
    def apply_noise(s, keys):
        noise = jax.random.normal(keys[4], s.shape) * 0.3
        return s + noise

    def apply_rotation(s, keys):
        theta = jax.random.uniform(keys[5], minval=-jnp.pi, maxval=jnp.pi)
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)

        key_dims = jax.random.split(keys[5])[0]
        dim_indices = jax.random.permutation(key_dims, n_species_dims)
        dim_a, dim_b = dim_indices[0], dim_indices[1]

        val_a = s[:, dim_a]
        val_b = s[:, dim_b]
        new_val_a = val_a * cos_theta - val_b * sin_theta
        new_val_b = val_a * sin_theta + val_b * cos_theta

        return s.at[:, dim_a].set(new_val_a).at[:, dim_b].set(new_val_b)

    def apply_scale(s, keys):
        scale = jax.random.uniform(keys[5], minval=0.5, maxval=1.5)
        return s * scale

    # Use jax.lax.switch to only compute the selected transformation
    subkeys_arr = jnp.stack(subkeys)
    species_transformed = jax.lax.switch(
        transform_type,
        [apply_noise, apply_rotation, apply_scale],
        species, subkeys_arr
    )

    # Apply transformation only to particles in circle, and only if mutating
    mask = in_circle[:, jnp.newaxis] & do_mutate
    species = jnp.where(mask, species_transformed, species)
    species = jnp.clip(species, -1.0, 1.0)

    # Also apply to alpha with independent noise (mask already includes do_mutate)
    alpha_noise = jax.random.normal(subkeys[4], alpha.shape) * 0.1
    alpha = jnp.where(mask, alpha + alpha_noise, alpha)
    alpha = jnp.clip(alpha, -1.0, 1.0)

    return species, alpha


@functools.partial(jax.jit, static_argnames=['max_steps', 'displacement_fn', 'shift_fn'])
def update_func_while(positions, velocities, species, alpha, neighbors, params, key, max_steps, box_size, displacement_fn, shift_fn):
    """Run simulation steps in a while loop until neighbor list overflows or max_steps reached.

    This is more efficient than fixed step counts because it maximizes the number of steps
    between neighbor list rebuilds.

    Returns:
        Updated positions, velocities, species, alpha, neighbors, key, and actual step count
    """

    def cond_fn(carry):
        """Continue while neighbor list hasn't overflowed and step count < max_steps."""
        _, _, neighbors, step_count = carry
        overflow_check = jnp.logical_not(neighbors.did_buffer_overflow)
        step_check = step_count < max_steps
        return jnp.logical_and(overflow_check, step_check)

    def body_fn(carry):
        """Run one simulation step."""
        x, v, neighbors, step_count = carry

        # Update neighbor list
        neighbors = neighbors.update(x)

        # Compute step (using precomputed damping)
        x, v = compute_step(
            x, v, species, alpha, neighbors.idx,
            params.mass, params.damping, params.dt,
            params.rmax, params.repulsion_dist, params.repulsion,
            displacement_fn, shift_fn
        )

        return x, v, neighbors, step_count + 1

    # Run while loop
    init_carry = (positions, velocities, neighbors, 0)
    positions, velocities, neighbors, final_step_count = jax.lax.while_loop(
        cond_fn, body_fn, init_carry
    )

    # Copy species using params from config
    key, *subkeys = jax.random.split(key, 3)
    species, alpha = copy_species_with_neighbors(
        subkeys, species, alpha, positions,
        neighbors.idx,
        max_copy_dist=params.max_copy_dist,
        max_species_dist=params.max_species_dist,
        copy_prob=params.copy_prob,
        displacement_fn=displacement_fn
    )

    # Apply circle-based mutation using params from config
    key, subkey = jax.random.split(key)
    species, alpha = mutate_circle(
        subkey, positions, species, alpha,
        box_size,
        mutation_prob=params.mutation_prob,
        displacement_fn=displacement_fn
    )

    return positions, velocities, species, alpha, neighbors, key, final_step_count


@functools.partial(jax.jit, static_argnames=['steps_per_selection', 'displacement_fn', 'shift_fn'])
def update_func(positions, velocities, species, alpha, neighbors, params, key, steps_per_selection, box_size, displacement_fn, shift_fn):
    """Original update function using fixed step count (kept for compatibility)."""
    # Run multiple physics sub-steps
    (positions, velocities, neighbors), _ = jax.lax.scan(
        functools.partial(multi_step, species=species, alpha=alpha, params=params, displacement_fn=displacement_fn, shift_fn=shift_fn),
        (positions, velocities, neighbors),
        None,
        length=steps_per_selection
    )

    # Copy species using params from config
    key, *subkeys = jax.random.split(key, 3)
    species, alpha = copy_species_with_neighbors(
        subkeys, species, alpha, positions,
        neighbors.idx,
        max_copy_dist=params.max_copy_dist,
        max_species_dist=params.max_species_dist,
        copy_prob=params.copy_prob,
        displacement_fn=displacement_fn
    )

    # Apply circle-based mutation using params from config
    key, subkey = jax.random.split(key)
    species, alpha = mutate_circle(
        subkey, positions, species, alpha,
        box_size,
        mutation_prob=params.mutation_prob,
        displacement_fn=displacement_fn
    )

    return positions, velocities, species, alpha, neighbors, key
        

class ParticleLife:
    def __init__(self, num_particles, species_dims, size=1.0, n_dims=2, dt=0.001, steps_per_frame=10):

        self.num_particles = num_particles
        self.species_dims = species_dims
        self.size = size
        self.n_dims = n_dims
        self.dt = dt
        self.steps_per_frame = steps_per_frame
        
        self.key = jax.random.PRNGKey(2)
        
        # Physics parameters
        self.mass = 0.02
        self.half_life = 0.001
        self.rmax = 0.2
        self.repulsion_dist = 0.05
        self.repulsion = 20.0

        # Mutation parameters (defaults, can be overridden via config)
        self.mutation_prob = 0.01
        self.max_copy_dist = 0.08
        self.max_species_dist = 0.2
        self.copy_prob = 0.001

        # Precompute damping coefficient
        self.damping = (0.5) ** (self.dt / self.half_life)

        self.params = Params(
            mass=self.mass,
            half_life=self.half_life,
            dt=self.dt,
            rmax=self.rmax,
            repulsion_dist=self.repulsion_dist,
            repulsion=self.repulsion,
            damping=self.damping,
            mutation_prob=self.mutation_prob,
            max_copy_dist=self.max_copy_dist,
            max_species_dist=self.max_species_dist,
            copy_prob=self.copy_prob,
        )
        
        # Create displacement function for periodic boundaries using jax-md
        self.displacement_fn, self.shift_fn = space.periodic(side=size)
        
        # Create neighbor list function from jax-md
        # We use the interaction radius as the cutoff
        self.neighbor_fn = partition.neighbor_list(
            self.displacement_fn,
            box=size,
            r_cutoff=self.rmax,
            dr_threshold=0.01,  # No buffer, rebuild every time
            capacity_multiplier=4.0,  # 300% extra capacity for safety
            format=partition.NeighborListFormat.Dense
        )
        
        # Initialize positions and species in tiles
        # Use tile size relative to rmax for consistent initialization across box sizes
        # Each tile is ~5x the interaction radius, ensuring good initial mixing
        positions = []
        species = []
        alpha = []
        tile_size = max(self.rmax * 5, 0.3)  # At least 0.3 to avoid too many tiles
        tiles_per_side = max(2, int(size[0] / tile_size))
        num_tiles = tiles_per_side ** n_dims

        # TODO set up for 3D
        for i in range(tiles_per_side):
            for j in range(tiles_per_side):
                start_x = size[0] * i / tiles_per_side
                start_y = size[1] * j / tiles_per_side
                self.key, *subkeys = jax.random.split(self.key, 6)
                
                particles_per_tile = num_particles // num_tiles
                pos_x = jax.random.uniform(subkeys[0], (particles_per_tile, 1), 
                                          minval=start_x, maxval=start_x + size[0] / tiles_per_side)
                pos_y = jax.random.uniform(subkeys[1], (particles_per_tile, 1), 
                                          minval=start_y, maxval=start_y + size[1] / tiles_per_side)
                if n_dims == 2:
                    positions.append(jnp.hstack([pos_x, pos_y]))
                elif n_dims == 3:
                    pos_z = jax.random.uniform(subkeys[2], (particles_per_tile, 1), 
                                              minval=0, maxval=size[2])
                    positions.append(jnp.hstack([pos_x, pos_y, pos_z]))

                species.append(jax.random.normal(subkeys[3], (1, species_dims)).repeat(particles_per_tile, axis=0))
                alpha.append(jax.random.uniform(subkeys[4], (1, species_dims), 
                                               minval=-0.2, maxval=0.2).repeat(particles_per_tile, axis=0))
        
        self.positions = jnp.vstack(positions)
        self.velocities = jnp.zeros_like(self.positions)
        self.species = jnp.vstack(species)
        self.alpha = jnp.vstack(alpha)
        
        # Allocate neighbor list
        self.neighbors = self.neighbor_fn.allocate(self.positions)
    
    def step(self):
        """Run a fixed number of simulation steps."""
        box_size = self.size[0] if hasattr(self.size, '__len__') else self.size

        self.positions, self.velocities, self.species, self.alpha, self.neighbors, self.key = update_func(
            self.positions, self.velocities, self.species, self.alpha, self.neighbors, self.params,
            self.key, self.steps_per_frame, box_size, self.displacement_fn, self.shift_fn
        )

        # Check if neighbor list overflowed
        if self.neighbors.did_buffer_overflow:
            print("Warning: Neighbor list overflow, reallocating...")
            self.neighbors = self.neighbor_fn.allocate(self.positions)

        return self.positions

    def step_while(self, max_steps=None):
        """Run simulation steps until neighbor list overflows or max_steps reached.

        This method is more efficient as it maximizes the number of steps between
        neighbor list rebuilds by using a JAX while loop.

        Args:
            max_steps: Maximum number of steps to run (default: 10x steps_per_frame)

        Returns:
            positions: Updated particle positions
            step_count: Number of steps actually executed
        """
        if max_steps is None:
            max_steps = self.steps_per_frame * 10

        box_size = self.size[0] if hasattr(self.size, '__len__') else self.size

        results = update_func_while(
            self.positions, self.velocities, self.species, self.alpha, self.neighbors, self.params,
            self.key, max_steps, box_size, self.displacement_fn, self.shift_fn
        )
        self.positions, self.velocities, self.species, self.alpha, self.neighbors, self.key, step_count = results

        # Check if neighbor list overflowed - if so, reallocate
        if self.neighbors.did_buffer_overflow:
            self.neighbors = self.neighbor_fn.allocate(self.positions)

        return self.positions, step_count


@jax.jit
def species_to_color(species):
    """Convert species vectors to RGB colors."""
    colors = jnp.stack([
        0.5 + 0.5 * jnp.maximum(species[:, 0], 0),
        0.5 + 0.5 * jnp.maximum(-species[:, 0], 0),
        0.5 + 0.5 * (species[:, 1] + 1) * 0.5,
    ], axis=-1)
    return colors


def main():
    # Simulation parameters
    num_particles = 4000
    species_dim = 2
    size = jnp.array([3.0, 3.0])
    n_dims = len(size)
    steps_per_frame = 20
    
    # Create simulation
    sim = ParticleLife(num_particles, species_dim, size, n_dims=n_dims, steps_per_frame=steps_per_frame)
    
    width, height = 800, 800

    render_to_screen = True
    # renderer = ParticleJAXRenderer(width, height, size, n_dims, num_particles, render_to_screen=render_to_screen, render_to_video=not render_to_screen)
    if render_to_screen:
        render_scale = 1.0  # Scale for display
        pygame.init()
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF, vsync=True)
        clock = pygame.time.Clock()
        while True:
            positions = sim.step()
            colours = species_to_color(sim.species)
            image = draw_particles_2d_fast(positions, colours, size, img_size=800)
            # Convert to numpy only for display (single transfer)
            image_np = np.array(image * 255).astype(np.uint8)

            # Scale up to display size
            surface = pygame.surfarray.make_surface(np.transpose(image_np, (1, 0, 2)))
            if render_scale != 1.0:
                surface = pygame.transform.scale(surface, (width, height))
            
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            
            clock.tick(60)
    else:
        import imageio.v2 as iio
        video_filename = "./outputs/evo_particle_life.mp4"
        num_frames = 1000
        w = iio.get_writer(video_filename, format='FFMPEG', mode='I', fps=30,
                       codec='libx264',
                       pixelformat='yuv420p')
        
        trajectory = []
        for frame_idx in tqdm(range(num_frames)):
            positions = sim.step()
            colours = species_to_color(sim.species)

            image = draw_particles_2d_fast(positions, colours, size, img_size=800)
        
            # Convert to numpy only for display (single transfer)
            image_np = np.array(image * 255).astype(np.uint8)
            w.append_data(image_np)

            # Store trajectory
            trajectory.append(positions)
        w.close()

        trajectories = jnp.stack(trajectory)
        audio = sonify_particles(trajectories, energy, force, species)

        print(f"Saved simulation video to {video_filename}")


if __name__ == "__main__":
    main()