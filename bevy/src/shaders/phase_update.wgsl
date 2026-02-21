@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: AudioParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.num_particles) { return; }

    let p = particles[id.x];
    let chunk_duration = f32(params.chunk_size) / params.sample_rate;

    // Advance phase by the same amount the audio shader would have used
    let end_phase = compute_phase(p, chunk_duration, params.energy_scale);
    particles[id.x].phase = end_phase % TAU;

    // Advance amp_phase for LFO
    let end_amp_phase = compute_amp_phase(p, chunk_duration, params.energy_scale);
    particles[id.x].amp_phase = end_amp_phase % TAU;
}