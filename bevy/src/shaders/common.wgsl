// Shared definitions for particle and audio shaders

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    phase: f32,
    energy: f32,
    species: vec2<f32>,
    alpha: vec2<f32>,
    interaction: vec2<f32>,
    amp_phase: f32,
    _pad: f32,
}

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// Spatial binning helpers
struct BinInfo {
    binIndex: u32,
    binX: u32,
    binY: u32,
}

fn getBinInfo(pos: vec2<f32>, map_x0: f32, map_y0: f32, bin_size: f32, num_bins_x: u32, num_bins_y: u32) -> BinInfo {
    let binX = u32((pos.x - map_x0) / bin_size);
    let binY = u32((pos.y - map_y0) / bin_size);
    let clampedX = clamp(binX, 0u, num_bins_x - 1u);
    let clampedY = clamp(binY, 0u, num_bins_y - 1u);
    return BinInfo(clampedY * num_bins_x + clampedX, clampedX, clampedY);
}

struct SimParams {
    dt: f32,
    time: f32,
    num_particles: u32,
    friction: f32,
    mass: f32,
    map_x0: f32,
    map_x1: f32,
    map_y0: f32,
    map_y1: f32,
    bin_size: f32,
    num_bins_x: u32,
    num_bins_y: u32,
    radius: f32,
    collision_radius: f32,
    collision_strength: f32,
    max_force_strength: f32,
    copy_radius: f32,
    copy_cos_sim_threshold: f32,
    copy_probability: f32,
    _pad: f32
}

fn normalize_energy(energy: f32, energy_scale: f32) -> f32 {
    return clamp(energy / energy_scale, -1.0, 1.0);
}

fn normalize_speed(speed: f32, max_speed: f32) -> f32 {
    return clamp(speed / max_speed, 0.0, 1.0);
}

// Waveform functions for audio synthesis

fn compute_frequency(p: Particle, t: f32, energy_scale: f32) -> f32 {
    // the less energy, the higher the frequency
    let energy_normalized = -1 * normalize_energy(p.energy, energy_scale) + 1.0;
    let base_freq = 80.0 + energy_normalized * 50.0;
    // TODO low pass filter if particles are further from centre of screen
    let mod1 = smoothstep(0.0, 1.0, p.species.x);
    let mod2 = smoothstep(0.0, 1.0, p.species.y);
    let mod3 = smoothstep(0.0, 1.0, -p.species.x);
    let mod4 = smoothstep(0.0, 1.0, -p.species.y);
    return base_freq + mod1 * sin(TAU * base_freq * 2.0 * t) + mod2 * sin(TAU * base_freq * 3.0 * t) + mod3 * sin(TAU * base_freq * 5.0 * t) + mod4 * sin(TAU * base_freq * 7.0 * t);
}

fn compute_amplitude(p: Particle, t: f32, max_speed: f32, energy_scale: f32) -> f32 {
    let speed = length(p.vel);
    // Normalize speed to 0-1 range
    let speed_normalized = normalize_speed(speed, max_speed);
    // Normalize energy to -1 to 1 range (negative = attractive/stable, positive = repulsive)
    let energy_normalized = normalize_energy(p.energy, energy_scale);
    // Lower energy = louder (invert so stable clusters are louder)
    // Faster movement = louder
    let energy_contrib = (1.0 - energy_normalized) * 0.5;  // 0 to 1
    let speed_contrib = speed_normalized;                   // 0 to 1
    return clamp(0.05 + energy_contrib * 0.35 + speed_contrib * 0.6, 0.0, 1.0);
}

fn compute_oscillator(phase: f32) -> f32 {
    return sin(phase);
}

fn compute_phase(p: Particle, t: f32, energy_scale: f32) -> f32 {
    let freq = compute_frequency(p, t, energy_scale);
    return p.phase + TAU * freq * t;
}

fn compute_amp_phase(p: Particle, t: f32, energy_scale: f32) -> f32 {
    // TODO particles with low energy should have complex and higher frequency LFOs
    let energy_normalized = (1.0 - normalize_energy(p.energy, energy_scale)) * 0.5;  // 0 to 1
    var lfo_freq = 0.005 + energy_normalized * (0.1 * p.interaction.x + 0.2 * sin(TAU * p.interaction.y * 0.1)); // base LFO frequency influenced by interaction
    lfo_freq = min(lfo_freq, 5.0); // maximum LFO frequency
    return p.amp_phase + TAU * lfo_freq * t;
}

struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
    current_x: vec2<f32>,  // Current viewport x (min, max)
    current_y: vec2<f32>,  // Current viewport y (min, max)
    map_x0: f32,
    map_y0: f32,
    bin_size: f32,
    num_bins_x: u32,
    num_bins_y: u32,
    max_speed: f32,        // Expected max particle speed for normalization
    energy_scale: f32,     // Expected energy scale for normalization
    _pad: u32,
}