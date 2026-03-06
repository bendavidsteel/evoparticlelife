// Render shader for GPU-side particle rendering
// Uses instancing to draw a quad per particle, fragment shader draws the particle

struct VertexInput {
    @location(0) position: vec2<f32>,  // Quad vertex position (-1 to 1)
    @location(1) uv: vec2<f32>,        // UV coordinates (0 to 1)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) core_hue: f32,
    @location(2) edge_hue: f32,
    @location(3) energy: f32,
    @location(4) vel: vec2<f32>,
    @location(5) alpha: vec2<f32>,
    @location(6) particle_id: f32,
}

struct RenderParams {
    map_x: vec2<f32>,
    map_y: vec2<f32>,
    current_x: vec2<f32>,
    current_y: vec2<f32>,
    particle_size: f32,
    num_particles: u32,
    max_speed: f32,
    energy_scale: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: RenderParams;
@group(0) @binding(2) var noise_texture: texture_2d<f32>;
@group(0) @binding(3) var noise_sampler: sampler;

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let particle = particles[instance];

    // Orthographic projection: map world coords to clip space [-1, 1]
    // Render in VIEWPORT space so trail texture is at full screen resolution
    // current_x.x = left, current_x.y = right
    // current_y.x = bottom, current_y.y = top
    let view_width = params.current_x.y - params.current_x.x;
    let view_height = params.current_y.y - params.current_y.x;

    // Transform particle position from world space to NDC (viewport space)
    let ndc_x = 2.0 * (particle.pos.x - params.current_x.x) / view_width - 1.0;
    let ndc_y = 2.0 * (particle.pos.y - params.current_y.x) / view_height - 1.0;

    let particle_size = params.particle_size * 3.0;

    // Scale particle size relative to viewport (particle_size is in world units)
    let size_ndc_x = particle_size * 2.0 / view_width;
    let size_ndc_y = particle_size * 2.0 / view_height;

    // Offset vertex by particle position in NDC
    let pos = vec2<f32>(
        ndc_x + vertex.position.x * size_ndc_x,
        ndc_y + vertex.position.y * size_ndc_y
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vertex.uv;
    out.core_hue = (particle.species.x + 1.0) * 0.5;  // Map [-1, 1] to [0, 1] for HSL
    out.edge_hue = (particle.species.y + 1.0) * 0.5;   // Second hue from species.y
    out.energy = particle.energy;
    out.vel = particle.vel;
    out.alpha = particle.alpha;
    out.particle_id = f32(instance);

    return out;
}

// HSL to RGB conversion
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - abs(h6 % 2.0 - 1.0));
    let m = l - c * 0.5;

    var rgb: vec3<f32>;
    if (h6 < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h6 < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h6 < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h6 < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h6 < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m);
}

// Hash function to generate pseudo-random UV offset per particle
fn hash_f32(x: f32) -> vec2<f32> {
    let s = vec2<f32>(x * 127.1, x * 311.7);
    return fract(sin(s) * 43758.5453123);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance from center of quad (UV is 0-1, center is 0.5)
    let centered_uv = in.uv - vec2<f32>(0.5);

    // Unstretched distance for circular core
    let dist = length(centered_uv) * 2.0;

    // Stretch glow opposite to velocity direction
    let speed = length(in.vel);
    let safe_speed = max(speed, 0.0001);
    let vel_dir = select(vec2<f32>(0.0, 1.0), in.vel / safe_speed, speed > 0.001);

    // Normalize speed (0 to 1 range)
    let speed_norm = min(speed / params.max_speed, 1.0);

    // How much is this pixel behind the particle (opposite to velocity)?
    let behind_amount = dot(centered_uv, -vel_dir);

    // Stretch factor: pixels behind the particle appear closer (expand glow backward)
    let stretch = 1.0 + max(behind_amount, 0.0) * speed_norm * 10.0;

    // Stretched distance for glow trail
    let glow_dist = dist / stretch;

    // Discard pixels outside the stretched glow
    if (glow_dist > 1.0) {
        discard;
    }

    // Energy-driven parameters (same as before)
    let scaled_energy = tanh(in.energy);
    let core_radius = 0.2 + 0.1 * scaled_energy;
    let energy_brightness = 0.8 + 0.2 * scaled_energy;

    // Smooth Gaussian core (replaces hard if/else)
    let core_sharpness = 8.0 + 4.0 * scaled_energy;
    let core_falloff = dist / core_radius;
    let core = exp(-core_sharpness * core_falloff * core_falloff);

    // Smooth membrane ring at core boundary
    let membrane_width = 0.15;
    let membrane = smoothstep(core_radius - membrane_width, core_radius, dist)
                  * smoothstep(core_radius + membrane_width, core_radius, dist);

    // Glow (unchanged from original)
    let glow = pow(1.0 - glow_dist, 2.0) * (0.5 + 0.2 * scaled_energy);

    // Sample blue noise for organic texture
    // Scale UV so each particle covers a visible patch of the noise texture
    // Offset by hashed particle_id so each particle gets a unique patch
    let noise_offset = hash_f32(in.particle_id);
    let noise_uv = in.uv * 64.0 + noise_offset * 512.0;
    let noise_val = textureSample(noise_texture, noise_sampler, noise_uv).r;

    // Noise modulation: visible variation in core brightness
    let noise_strength = 0.3 + 0.15 * abs(in.alpha.x);
    let noise_mod = 1.0 - noise_strength + noise_strength * noise_val;

    // Dual hue: interpolate from core color to edge color based on distance
    let hue_blend = smoothstep(0.0, core_radius * 2.0, dist);
    // Alpha.y shifts the blend curve for per-particle variation
    let hue_t = clamp(hue_blend + in.alpha.y * 0.2, 0.0, 1.0);
    let core_rgb = hsl_to_rgb(in.core_hue, 0.9, energy_brightness * 0.5);
    let edge_rgb = hsl_to_rgb(in.edge_hue, 0.7, energy_brightness * 0.6);
    let rgb = mix(core_rgb, edge_rgb, hue_t);

    // Combine core, membrane, and glow with noise modulation
    let combined = max(core * 0.5 * noise_mod + membrane * 0.4, glow);
    let alpha = combined * 0.9;

    return vec4<f32>(rgb * combined * noise_mod, alpha);
}
