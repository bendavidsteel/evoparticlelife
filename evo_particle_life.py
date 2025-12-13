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
])

@jax.jit
def force_graph(r, rmax, alpha, repulsion_dist, repulsion):
    first = jnp.maximum(repulsion_dist - r, 0.) * -repulsion
    second = alpha * jnp.maximum(1 - abs(2 * r - rmax- repulsion_dist) / (rmax - repulsion_dist), 0)
    cond_first = (r < repulsion_dist)
    cond_second = (repulsion_dist < r) & (r < rmax)
    return jnp.where(cond_first, first, jnp.where(cond_second, second, 0.))


@functools.partial(jax.jit, static_argnames=['displacement_fn'])
def compute_forces_with_neighbors(positions, species, alpha, neighbor_idx, rmax, repulsion_dist, repulsion, displacement_fn):
    """Compute forces using jax-md neighbor lists.
    
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
    N = positions.shape[0]
    n_dims = positions.shape[1]
    
    def compute_particle_force(pos_i, alpha_i, neighbors):
        """Compute force on particle i from its neighbors."""
        def _force_from_neighbor(n_x, n_s):
            # Compute displacement using jax-md's function (handles periodic boundaries)
            dr = displacement_fn(n_x, pos_i)
            r = jnp.sqrt(jnp.sum(jnp.square(dr)).clip(1e-10))
            
            # Compute force
            interaction = jnp.dot(n_s, alpha_i)
            force_scalar = force_graph(r, rmax, interaction, repulsion_dist, repulsion)
            
            direction = dr / (r + 1e-10)
            force_vec = direction * force_scalar
            
            return force_vec
        
        def force_from_neighbor(j, n_x, n_s):
            # Check if valid neighbor (jax-md uses N to indicate padding)
            return jax.lax.cond(j != N, lambda: _force_from_neighbor(n_x, n_s), lambda: jnp.zeros(n_dims))

        # Vectorize over neighbors
        n_xs, n_species = positions[neighbors], species[neighbors]
        forces_from_neighbors = jax.vmap(force_from_neighbor)(neighbors, n_xs, n_species)
        return forces_from_neighbors.sum(axis=0)
    
    # Vectorize over all particles
    forces = jax.vmap(compute_particle_force)(positions, alpha, neighbor_idx)
    return forces


@functools.partial(jax.jit, static_argnames=['displacement_fn', 'shift_fn'])
def compute_step(x, v, species, alpha, neighbor_idx, mass, half_life, dt, rmax, 
                 repulsion_dist, repulsion, displacement_fn, shift_fn):
    """Compute simulation step using jax-md neighbor lists.
    
    Args:
        x: (N, 2) positions
        v: (N, 2) velocities
        species: (N, species_dims) species vectors
        alpha: (N, species_dims) interaction matrix
        neighbor_idx: (N, max_neighbors) neighbor indices
        mass: scalar mass
        half_life: scalar half-life
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
    
    # Update velocities and positions
    acc = f / mass
    mu = (0.5) ** (dt / half_life)
    v = mu * v + acc * dt
    x = shift_fn(x, v * dt)
    
    return x, v

@functools.partial(jax.jit, static_argnames=['displacement_fn'])
def copy_species_with_neighbors(subkeys, species, alpha, positions, neighbor_idx, 
                                max_copy_dist=0.08, max_species_dist=0.2, copy_prob=0.001,
                                displacement_fn=None):
    """Copy species using jax-md neighbor lists.
    
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
    
    def compute_copy_probs(i, pos_i, species_i, neighbors):
        """Compute copy probabilities for particle i from its neighbors."""
        
        def prob_from_neighbor(j, n_x, n_s):
            # Check if valid neighbor
            return jax.lax.cond(
                (j != N) & (j != i),
                lambda: _prob_from_neighbor(j, n_x, n_s),
                lambda: (0.0, j)
            )

        def _prob_from_neighbor(j, n_x, n_s):
            # Compute distance
            dr = displacement_fn(n_x, pos_i)
            r = jnp.sqrt(jnp.sum(dr ** 2).clip(1e-10))
            
            # Compute species difference
            species_diff = jnp.linalg.norm(n_s - species_i)
            
            # Check conditions
            within_dist = r < max_copy_dist
            within_species = species_diff < max_species_dist
            valid = within_dist & within_species
            
            # Compute probability
            dist_factor = 1.0 - r / max_copy_dist
            species_factor = jnp.pow(10.0, -species_diff)
            prob = jnp.where(valid, dist_factor * species_factor, 0.0)
            
            return prob, j
        
        # Vectorize over neighbors
        n_positions, n_species = positions[neighbors], species[neighbors]
        probs, indices = jax.vmap(prob_from_neighbor)(neighbors, n_positions, n_species)
        return probs, indices
    
    # Get copy probabilities for all particles
    all_probs, all_indices = jax.vmap(compute_copy_probs)(jnp.arange(N), positions, species, neighbor_idx)
    
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
    
    # Compute step
    x, v = compute_step(
        x, v, species, alpha, neighbors.idx,
        params.mass, params.half_life, params.dt,
        params.rmax, params.repulsion_dist, params.repulsion,
        displacement_fn, shift_fn
    )

    return (x, v, neighbors), x

@functools.partial(jax.jit, static_argnames=['steps_per_selection', 'displacement_fn', 'shift_fn'])
def update_func(positions, velocities, species, alpha, neighbors, params, key, steps_per_selection, displacement_fn, shift_fn):
    # Run multiple physics sub-steps
    (positions, velocities, neighbors), _ = jax.lax.scan(
        functools.partial(multi_step, species=species, alpha=alpha, params=params, displacement_fn=displacement_fn, shift_fn=shift_fn),
        (positions, velocities, neighbors),
        None,
        length=steps_per_selection
    )
    
    # Copy species
    key, *subkeys = jax.random.split(key, 3)
    species, alpha = copy_species_with_neighbors(
        subkeys, species, alpha, positions,
        neighbors.idx, displacement_fn=displacement_fn
    )
    
    # Apply mutation
    mutation_rate = 1e-6
    key, subkey = jax.random.split(key)
    mutation_mask = jax.random.uniform(subkey, species.shape) < mutation_rate
    mutation_values = jax.random.normal(subkey, species.shape) * 0.1
    species = jnp.where(mutation_mask, species + mutation_values, species)
    species = jnp.clip(species, -1.0, 1.0)

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
        
        # Parameters
        self.mass = 0.02
        self.half_life = 0.001
        self.rmax = 0.2
        self.repulsion_dist = 0.05
        self.repulsion = 20.0

        self.params = Params(
            mass=self.mass,
            half_life=self.half_life,
            dt=self.dt,
            rmax=self.rmax,
            repulsion_dist=self.repulsion_dist,
            repulsion=self.repulsion,
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
        positions = []
        species = []
        alpha = []
        tiles_per_side = int(size[0] / 0.5)
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
        self.positions, self.velocities, self.species, self.alpha, self.neighbors, self.key = update_func(
            self.positions, self.velocities, self.species, self.alpha, self.neighbors, self.params,
            self.key, self.steps_per_frame, self.displacement_fn, self.shift_fn
        )

        # Check if neighbor list overflowed
        if self.neighbors.did_buffer_overflow:
            print("Warning: Neighbor list overflow, reallocating...")
            self.neighbors = self.neighbor_fn.allocate(self.positions)

        return self.positions


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

    render_to_screen = False
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