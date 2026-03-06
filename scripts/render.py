import collections
import functools

import einops
import jax
import jax.numpy as jp


@functools.partial(jax.jit, static_argnames=['img_size'])
def draw_particles_2d_fast(positions, particle_colors, map_size, img_size=800):
    # Use explicit float32 throughout for consistency regardless of x64 mode
    image = jp.zeros((img_size, img_size, 3), dtype=jp.float32)

    splat_size = 10
    linspace = jp.linspace(-1, 1, splat_size, dtype=jp.float32)
    gaussian = jp.exp(-linspace**2 / 0.1)
    splat_template = jp.outer(gaussian, gaussian)

    # Ensure colors are float32
    particle_colors_f32 = particle_colors.astype(jp.float32)
    splats = splat_template[..., jp.newaxis] * particle_colors_f32[:, jp.newaxis, jp.newaxis, :]
    position_coords = (positions * img_size / map_size).astype(jp.int32)

    def add_splat(image, inputs):
        position, splat = inputs
        # Extract the region, add splat, then update
        y_start = position[1] - splat_size // 2
        x_start = position[0] - splat_size // 2
        # Use explicit int32 for channel index to match position dtype
        zero = jp.int32(0)

        current_region = jax.lax.dynamic_slice(
            image, (y_start, x_start, zero), (splat_size, splat_size, 3)
        )
        updated_region = current_region + splat

        image = jax.lax.dynamic_update_slice(image, updated_region, (y_start, x_start, zero))
        return image, None

    image, _ = jax.lax.scan(add_splat, image, (position_coords, splats))
    return jp.clip(image, 0, 1)
    

@functools.partial(jax.jit, static_argnames=['map_size', 'num_species', 'start', 'offset'])
def draw_multi_species_particles(trajectory, map_size, species, num_species, 
                                   start=-16000, offset=2000):
    """
    Wrapper to create three orthogonal view animations with colored species for 3D particles.
    """
    # Create a color map for different species
    angles = jp.linspace(0, 2 * jp.pi, num_species, endpoint=False)
    colors = jp.stack([
        jp.sin(angles) * 0.5 + 0.5,
        jp.sin(angles + 2 * jp.pi / 3) * 0.5 + 0.5,
        jp.sin(angles + 4 * jp.pi / 3) * 0.5 + 0.5
    ], axis=1)
    
    # Pre-compute particle colors based on species
    particle_colors = colors[species]
    
    return _draw_particles(trajectory, map_size, particle_colors, 
                                  start=start, offset=offset)


@functools.partial(jax.jit, static_argnames=('map_size', 'img_size', 'particle_radius', 'start', 'offset'))
def draw_particles_3d_views(trajectory, map_size, particle_colors, img_size=800, 
                            particle_radius=5, start=-16000, offset=2000):
    """
    Create three animations of 3D particle trajectories from orthogonal viewing angles.
    
    Args:
        trajectory: JAX array of shape (num_frames, num_particles, 3)
                   containing 3D particle positions at each time step
        map_size: Size of the simulation space (assumed cubic)
        particle_colors: Colors for each particle of shape (num_particles, 3)
        img_size: Size of the output images (square)
        particle_radius: Radius of particles in pixels
        start: Starting frame index
        offset: Frame sampling interval
        
    Returns:
        Tuple of three video arrays (xy_view, xz_view, yz_view)
    """
    # Subsample frames
    subsampled_trajectory = trajectory[start::offset]
    num_frames = subsampled_trajectory.shape[0]
    num_particles = subsampled_trajectory.shape[1]
    
    # Create three trajectory sets for different orthogonal projections
    # XY projection (top view)
    xy_view_traj = subsampled_trajectory  # Take just X and Y coordinates
    
    # XZ projection (side view)
    xz_view_traj = jp.stack([
        subsampled_trajectory[:, :, 0],  # X coordinate
        subsampled_trajectory[:, :, 2],  # Z coordinate
        subsampled_trajectory[:, :, 1]
    ], axis=-1)
    
    # YZ projection (front view)
    yz_view_traj = jp.stack([
        subsampled_trajectory[:, :, 1],  # Y coordinate
        subsampled_trajectory[:, :, 2],  # Z coordinate
        subsampled_trajectory[:, :, 0]
    ], axis=-1)
    
    # Define the rendering function
    @jax.jit
    def render_frame(positions):
        """
        Render particles using a splatting technique.
        """
        # Scale positions to image coordinates
        positions, depth = positions[..., :2], positions[..., 2]
        idx_order = jp.argsort(depth, descending=True)
        positions, depth = positions[idx_order, :], depth[idx_order]
        scaled_pos = (positions / map_size) * img_size
        scaled_pos = jp.clip(scaled_pos, 0, img_size - 1)
        
        # Start with a black background
        image = jp.zeros((img_size, img_size, 3))
        
        # Create a grid of coordinates
        x_indices = jp.arange(img_size)
        y_indices = jp.arange(img_size)
        X, Y = jp.meshgrid(x_indices, y_indices)
        coords = jp.stack([X, Y], axis=-1)  # Shape: (img_size, img_size, 2)
        
        # For each particle, compute influence on each pixel
        def process_particle(image, particle_idx):
            pos = scaled_pos[particle_idx]
            this_depth = depth[particle_idx]
            color = particle_colors[particle_idx]
            
            # Calculate distance from each pixel to the particle
            dist_squared = jp.sum((coords - pos)**2, axis=-1)
            this_particle_radius = particle_radius * (1 - 0.5 * this_depth / map_size)

            # Create a mask for pixels affected by this particle
            mask = dist_squared <= (this_particle_radius**2)
            mask = mask[:, :, jp.newaxis]  # Add channel dimension

            this_color = (1 - 0.5 * this_depth / map_size) * color + (0.5 * this_depth / map_size) * jp.zeros((3,))
            
            # Update image: where mask is True, use particle color
            return jp.where(mask, this_color, image)
        
        # Scan through all particles
        image, _ = jax.lax.scan(
            lambda img, idx: (process_particle(img, idx), None),
            image,
            jp.arange(num_particles)
        )
        
        return (image * 255).astype(jp.uint8)
    
    # Render all frames for each view using vmap
    xy_frames = jax.vmap(render_frame)(xy_view_traj)
    xz_frames = jax.vmap(render_frame)(xz_view_traj)
    yz_frames = jax.vmap(render_frame)(yz_view_traj)
    
    return jp.stack([xy_frames, xz_frames, yz_frames])

def draw_multi_species_particles_3d(trajectory, map_size, species=None, num_species=None, 
                                   start=-16000, offset=2000):
    """
    Wrapper to create three orthogonal view animations with colored species for 3D particles.
    """
    # Create a color map for different species
    angles = jp.linspace(0, 2 * jp.pi, num_species, endpoint=False)
    colors = jp.stack([
        jp.sin(angles) * 0.5 + 0.5,
        jp.sin(angles + 2 * jp.pi / 3) * 0.5 + 0.5,
        jp.sin(angles + 4 * jp.pi / 3) * 0.5 + 0.5
    ], axis=1)
    
    # Pre-compute particle colors based on species
    particle_colors = colors[species]
    
    return draw_particles_3d_views(trajectory, map_size, particle_colors, 
                                  start=start, offset=offset)

def draw_particles_3d(trajectory, map_size, start=-16000, offset=2000):
    """
    Wrapper to create three orthogonal view animations with default particle colors for 3D particles.
    """
    particle_colors = jp.zeros((trajectory.shape[1], 3))
    return draw_particles_3d_views(trajectory, map_size, particle_colors, 
                                  start=start, offset=offset)

def generate_colors(n):
    """
    Generate n discrete colors with maximum saturation and value in HSV.
    
    Parameters:
    n (int): Number of colors to generate
    
    Returns:
    list: List of RGB color tuples
    """
    rgb_colors = cc.glasbey_bw_minc_20[:n]
    
    return jp.asarray(rgb_colors)

@functools.partial(jax.jit, static_argnames=['map_size', 'start', 'offset'])
def fancy_draw_particles(trajectory, energies, map_size, species, colors, 
                                   start=-16000, offset=2000):
    """
    Wrapper to create three orthogonal view animations with colored species for 3D particles.
    """
    # Pre-compute particle colors based on species
    particle_colors = colors[species]
    
    subsampled_trajectory = trajectory[start::offset]
    
    # Define image dimensions and particle rendering parameters
    img_size = 800
    particle_radius = 10

    @jax.jit
    def hit_color_func(dists, radii, coord):
        min_idx = jp.argmin(dists)
        min_dist = dists[min_idx]
        min_radius = radii[min_idx] / 2
        bloom_color = particle_colors[min_idx]
        # should be 0 when dist == radius, 1 when dist = 0
        intensity = jp.maximum((1 - jp.clip(min_dist / min_radius, 0, 1) ** 3), jp.exp(-min_dist / min_radius))
        return bloom_color * intensity

    @jax.jit
    def no_hit_color_func(dists, radii, coord):
        return jp.zeros(3)
    
    # Alternative approach: use a splatting technique
    @jax.jit
    def render_frame(positions, energy):
        """
        Render particles using a splatting technique.
        
        This creates an alpha mask for each particle and composites them onto the image.
        """
        # Scale positions to image coordinates
        scaled_pos = (positions / map_size) * img_size
        scaled_pos = jp.clip(scaled_pos, 0, img_size - 1)

        radii = (particle_radius + 5 * jp.exp(-energy)) * 2

        # Create a grid of coordinates
        x_indices = jp.arange(img_size)
        y_indices = jp.arange(img_size)
        X, Y = jp.meshgrid(x_indices, y_indices)
        coords = jp.stack([X, Y], axis=-1)  # Shape: (img_size, img_size, 2)
        
        # for each pixel, compute color
        def process_pixel(coord):
            dists = jp.sqrt(jp.sum((coord - scaled_pos) ** 2, axis=-1))
            min_dist_idx = jp.argmin(dists)
            min_dist = dists[min_dist_idx]
            min_dist_radius = radii[min_dist_idx]

            return jax.lax.cond(min_dist <= min_dist_radius, hit_color_func, no_hit_color_func, dists, radii, coord)

        image_fn = jax.vmap(jax.vmap(process_pixel, in_axes=0), in_axes=1)
        image = image_fn(coords)
        
        return (image * 255).astype(jp.uint8)
    
    frames = jax.vmap(render_frame)(subsampled_trajectory, energies)

    return frames

@functools.partial(jax.jit, static_argnames=('fps', 'sample_rate'))
def resample(x, fps, sample_rate):
    return jp.interp(jp.arange(0, x.shape[0], fps / sample_rate), jp.arange(x.shape[0]), x)

@functools.partial(jax.jit, static_argnames=('cutoff_freq', 'sample_rate'))
def simple_lowpass(signal, cutoff_freq, sample_rate):
    # Simple one-pole lowpass filter
    rc = 1.0 / (2 * jp.pi * cutoff_freq)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)
    
    def scan_fn(carry, x):
        y = alpha * x + (1 - alpha) * carry
        return y, y
    
    _, filtered = jax.lax.scan(scan_fn, 0.0, signal)
    return filtered

def soft_clip(x, threshold=0.8):
    # Soft clipping function
    return jp.where(
        jp.abs(x) <= threshold,
        x,
        threshold * jp.tanh(x / threshold)
    )

def create_envelope(length, attack_samples=1000):
    attack = jp.linspace(0, 1, attack_samples)
    sustain = jp.ones(length - attack_samples)
    return jp.concatenate([attack, sustain])

def triangle(x, center=0.5):
    y1 = jp.clip((1 / center) * x, 0, 1)
    y2 = jp.clip(1 / (1 - center) - (1 / (1 - center)) * x, 0, 1)
    return 2 * y1 * y2 - 1

# @functools.partial(jax.jit, static_argnames=('fps', 'sample_rate'))
def sonify_particles(trajectory, energy, force, species, num_species, map_size, fps=30, sample_rate=44100):
    base_freq = 100
    resampled_energy = jax.vmap(resample, in_axes=(1, None, None))(energy, fps, sample_rate)

    # Add vibrato to frequency before generating waveform
    vibrato_rate = 5.0  # Hz
    vibrato_depth = 0.01  # 1% of frequency
    time = jp.arange(resampled_energy.shape[1]) / sample_rate
    vibrato = 1 + vibrato_depth * jp.sin(2 * jp.pi * vibrato_rate * time)

    freq = (400 * jp.exp(-resampled_energy) + base_freq) * vibrato[None, :]
    p = jax.vmap(modcumsum, in_axes=(0, None))(freq / sample_rate, 1)
    species_skew = 0.01 + species * 0.1
    sawtooth = jax.vmap(triangle)(p, center=species_skew)
    signal = jax.vmap(simple_lowpass, in_axes=(0, None, None))(sawtooth, 1000, sample_rate)

    centre = jp.array([map_size / 2, map_size / 2])
    left_ear = centre - jp.array([map_size / 3, 0])
    right_ear = centre + jp.array([map_size / 3, 0])
    left_ear_dist = jp.linalg.norm(trajectory - left_ear, axis=-1)
    right_ear_dist = jp.linalg.norm(trajectory - right_ear, axis=-1)
    
    audio_unit_dist = 2
    min_dist = 1.0  # Prevent division by very small numbers
    force_mag = jp.linalg.norm(force, axis=-1)

    left_gain = jp.log1p(force_mag) / jp.maximum(left_ear_dist / audio_unit_dist ** 2, min_dist)
    right_gain = jp.log1p(force_mag) / jp.maximum(right_ear_dist / audio_unit_dist ** 2, min_dist)

    left_gain = jax.vmap(resample, in_axes=(1, None, None))(left_gain, fps, sample_rate)
    right_gain = jax.vmap(resample, in_axes=(1, None, None))(right_gain, fps, sample_rate)

    # Apply gains with softer limiting
    left_channel = signal * left_gain
    right_channel = signal * right_gain

    # Soft clipping with RMS normalization
    num_particles = trajectory.shape[1]
    left_channel = soft_clip(left_channel.sum(axis=0) / jp.sqrt(num_particles)) * 0.8
    right_channel = soft_clip(right_channel.sum(axis=0) / jp.sqrt(num_particles)) * 0.8

    envelope = create_envelope(signal.shape[1], int(0.1 * sample_rate))  # 100ms attack
    left_channel *= envelope
    right_channel *= envelope

    return jp.stack([
        left_channel,
        right_channel
    ], axis=0)


@jax.jit
def modcumsum(x, m):
    assert len(x.shape) == 1
    def modsum(carry, x):
        carry = (carry + x) % m
        return carry, carry
    _, y = jax.lax.scan(modsum, 0, x)
    return y


def chunk_sonify_particles(trajectory, energy, force, fps=30):
    num_frames = trajectory.shape[0]
    sample_rate = 44100
    duration = num_frames / fps  # total duration in seconds
    t = jp.arange(0, duration, 1 / sample_rate)
    base_freq = 200

    left_channel = jp.zeros_like(t)
    right_channel = jp.zeros_like(t)

    # sawtooth wave for each particle
    return jp.stack([
        left_channel,
        right_channel
    ], axis=-1).astype(jp.float32)  # shape (num_samples, 2)

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt'))
def multi_step_scan(params, initial_points, species, dt, num_steps, map_size, periodic=True):
    def scan_step(carry, _):
        points = carry
        points = step_f(params, points, species, dt, map_size, periodic=periodic)
        return points, points

    final_points, trajectory = jax.lax.scan(
        scan_step,
        initial_points,
        xs=None,
        length=num_steps
    )

    return final_points, trajectory

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt'))
def evo_multi_step_scan(params, initial_particles, species, dt, num_steps, map_size, key):
    def scan_step(carry, step_key):
        params_t, particles, species_t = carry
        new_params, new_particles, new_species = evo_step_f(params_t, particles, species_t, dt, map_size, step_key)
        return (new_params, new_particles, new_species), new_particles

    # Generate keys for each step
    step_keys = jax.random.split(key, num_steps)
    
    (final_params, final_particles, final_species), trajectory = jax.lax.scan(
        scan_step,
        (params, initial_particles, species),
        step_keys
    )

    return (final_params, final_particles, final_species), trajectory

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt', 'map_size'))
def multi_step_scan_with_force(params, initial_points, species, dt, num_steps, map_size):
    def scan_step(carry, _):
        points = carry
        f = direct_motion_f(params, points, species, map_size)
        points += dt * f
        points -= jp.floor(points/map_size) * map_size  # periodic boundary
        return points, (points, f)

    final_points, (trajectory, force) = jax.lax.scan(
        scan_step,
        initial_points,
        xs=None,
        length=num_steps
    )

    return final_points, (trajectory, force)

@functools.partial(jax.jit, static_argnames=('num_steps', 'dt', 'map_size'))
def multi_step_scan_with_force_and_energy(params, initial_points, species, dt, num_steps, map_size, periodic=True):
    def scan_step(carry, _):
        points = carry
        f, e = motion_and_energy_f(params, points, species, map_size, periodic=periodic)
        points += dt * f
        points = jax.lax.cond(
            periodic,
            lambda x: x - jp.floor(x/map_size) * map_size, # periodic boundary
            lambda x: jp.clip(x, 0, map_size),
            points
        )
        return points, (points, f, e)

    final_points, (trajectory, force, energy) = jax.lax.scan(
        scan_step,
        initial_points,
        xs=None,
        length=num_steps
    )

    return final_points, (trajectory, force, energy)
