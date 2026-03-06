// Fullscreen shader utilities
// Common vertex shader for fullscreen passes, plus various fragment shaders

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle vertices (3 vertices cover the screen)
@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate fullscreen triangle from vertex index
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Reproject pass: sample previous trail texture with viewport reprojection and decay
// Each frame, the old trail content is warped to account for camera pan/zoom,
// then decayed. This keeps trails stable in world space while rendering at full resolution.

struct ReprojectParams {
    prev_x: vec2<f32>,     // Previous viewport x (min, max)
    prev_y: vec2<f32>,     // Previous viewport y (min, max)
    current_x: vec2<f32>,  // Current viewport x (min, max)
    current_y: vec2<f32>,  // Current viewport y (min, max)
    decay: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var prev_texture: texture_2d<f32>;
@group(0) @binding(1) var prev_sampler: sampler;
@group(0) @binding(2) var<uniform> reproject_params: ReprojectParams;

@fragment
fn fs_reproject(in: VertexOutput) -> @location(0) vec4<f32> {
    // Convert current screen UV to world coordinates using current viewport
    let world_x = reproject_params.current_x.x + in.uv.x * (reproject_params.current_x.y - reproject_params.current_x.x);
    let world_y = reproject_params.current_y.y - in.uv.y * (reproject_params.current_y.y - reproject_params.current_y.x);

    // Convert world coordinates to previous frame's screen UV
    let prev_u = (world_x - reproject_params.prev_x.x) / (reproject_params.prev_x.y - reproject_params.prev_x.x);
    let prev_v = (reproject_params.prev_y.y - world_y) / (reproject_params.prev_y.y - reproject_params.prev_y.x);

    // Out of bounds = black (no trail data in newly revealed areas)
    if (prev_u < 0.0 || prev_u > 1.0 || prev_v < 0.0 || prev_v > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // Sample previous frame and apply decay
    let prev_color = textureSample(prev_texture, prev_sampler, vec2<f32>(prev_u, prev_v));
    return vec4<f32>(prev_color.rgb * reproject_params.decay, 1.0);
}

// Blit pass: simple passthrough (trail texture is already in viewport space)
@group(0) @binding(0) var blit_texture: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

@fragment
fn fs_blit(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(blit_texture, blit_sampler, in.uv);
}
