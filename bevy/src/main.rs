use nannou::prelude::*;
use nannou::wgpu::BufferInitDescriptor;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;
use image::ImageReader;

#[cfg(not(target_arch = "wasm32"))]
use nannou_audio as audio;
#[cfg(not(target_arch = "wasm32"))]
use nannou_audio::Buffer;
#[cfg(not(target_arch = "wasm32"))]
use std::thread::JoinHandle;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicI8;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use crossbeam_channel::{self, Receiver};
#[cfg(not(target_arch = "wasm32"))]
use ringbuf::{traits::{Consumer, Producer, Split, Observer}, HeapRb};

const NUM_PARTICLES: u32 = 8192;
#[cfg(not(target_arch = "wasm32"))]
const NUM_CHANNELS: u32 = 2;
#[cfg(not(target_arch = "wasm32"))]
const SAMPLE_RATE: u32 = 44100;
#[cfg(not(target_arch = "wasm32"))]
const CHUNK_SIZE: u32 = 2048;
#[cfg(not(target_arch = "wasm32"))]
const NUM_AUDIO_STAGING_BUFS: usize = 4; // Triple-buffered to avoid "buffer still mapped" race
#[cfg(not(target_arch = "wasm32"))]
const CHUNK_FLOATS: u32 = CHUNK_SIZE * NUM_CHANNELS;
#[cfg(not(target_arch = "wasm32"))]
const INITIAL_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 2;
#[cfg(not(target_arch = "wasm32"))]
const MIN_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS;
#[cfg(not(target_arch = "wasm32"))]
const MAX_REQUEST_THRESHOLD: u32 = CHUNK_FLOATS * 4;
#[cfg(not(target_arch = "wasm32"))]
const BUFFER_HISTORY_SIZE: usize = 64;
const PARTICLE_SIZE: f32 = 0.05;
const BIN_SIZE: f32 = 0.5;
const PAN_ACCEL: f32 = 0.015;
const PAN_MAX_SPEED: f32 = 0.04;
const PAN_FRICTION: f32 = 0.85;
const ZOOM_ACCEL: f32 = 0.004;
const ZOOM_MAX_SPEED: f32 = 0.02;
const ZOOM_FRICTION: f32 = 0.85;
#[cfg(not(target_arch = "wasm32"))]
const AUDIO_LOG_INTERVAL_FRAMES: u64 = 60; // Log audio stats every ~1 second at 60fps

// Shader sources (loaded at compile time)
const PARTICLE_SHADER: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/particle.wgsl"));
#[cfg(not(target_arch = "wasm32"))]
const AUDIO_SHADER: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/audio.wgsl"));
const RENDER_SHADER: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/render.wgsl"));
#[cfg(not(target_arch = "wasm32"))]
const PHASE_UPDATE_SHADER: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/phase_update.wgsl"));
const BIN_FILL_SIZE_SHADER: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/bin_fill_size.wgsl"));
const BIN_PREFIX_SUM_SHADER: &str = include_str!("shaders/bin_prefix_sum.wgsl");
const PARTICLE_SORT_SHADER: &str = concat!(include_str!("shaders/common.wgsl"), include_str!("shaders/particle_sort.wgsl"));
const FULLSCREEN_SHADER: &str = include_str!("shaders/fullscreen.wgsl");

// Particle struct - must match shader layout exactly
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    phase: f32,
    energy: f32,
    species: [f32; 2],
    alpha: [f32; 2],
    interaction: [f32; 2],
    amp_phase: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AudioParams {
    sample_rate: f32,
    num_particles: u32,
    chunk_size: u32,
    volume: f32,
    current_x: [f32; 2],  // Current viewport x (min, max)
    current_y: [f32; 2],  // Current viewport y (min, max)
    map_x0: f32,
    map_y0: f32,
    bin_size: f32,
    num_bins_x: u32,
    num_bins_y: u32,
    max_speed: f32,       // Expected max particle speed for normalization
    energy_scale: f32,    // Expected energy scale for normalization
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderParams {
    map_x: [f32; 2],      // Simulation bounds x (min, max)
    map_y: [f32; 2],      // Simulation bounds y (min, max)
    current_x: [f32; 2],  // Current viewport x (min, max)
    current_y: [f32; 2],  // Current viewport y (min, max)
    particle_size: f32,
    num_particles: u32,
    max_speed: f32,
    energy_scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReprojectParams {
    prev_x: [f32; 2],     // Previous viewport x (min, max)
    prev_y: [f32; 2],     // Previous viewport y (min, max)
    current_x: [f32; 2],  // Current viewport x (min, max)
    current_y: [f32; 2],  // Current viewport y (min, max)
    decay: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

// Quad vertex for instanced rendering
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

// Unit quad vertices (will be scaled by particle size in shader)
const QUAD_VERTICES: [QuadVertex; 4] = [
    QuadVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    QuadVertex { position: [ 1.0, -1.0], uv: [1.0, 1.0] },
    QuadVertex { position: [-1.0,  1.0], uv: [0.0, 0.0] },
    QuadVertex { position: [ 1.0,  1.0], uv: [1.0, 0.0] },
];

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 1, 3, 2];

// Static vertex attributes for the quad
const QUAD_VERTEX_ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct AudioFeedback {
    min_buffer_level: u32,  // Minimum buffer level over recent history
    current_buffer_level: u32,
}

#[derive(Clone)]
struct SpatialHash {
    // Buffers for spatial hashing
    bin_size_buf: Arc<wgpu::Buffer>,
    bin_offset_buf: Arc<wgpu::Buffer>,          // Result of prefix sum
    bin_offset_tmp_buf: Arc<wgpu::Buffer>,      // Temp buffer for prefix sum ping-pong
    prefix_sum_step_buf: Arc<wgpu::Buffer>,     // Uniform for step size

    // Bind groups for fill_bin_size - dual for ping-pong [A, B]
    // Note: clear_bin_size uses the same bind groups
    fill_bin_size_particles_bind_groups: [Arc<wgpu::BindGroup>; 2],
    fill_bin_size_params_bind_group: Arc<wgpu::BindGroup>,
    fill_bin_size_bins_bind_group: Arc<wgpu::BindGroup>,
    // Bind groups for prefix sum (ping-pong: 0=size->offset, 1=offset->tmp, 2=tmp->offset)
    prefix_sum_bind_groups: [Arc<wgpu::BindGroup>; 3],
    // Bind groups for sort particles - dual for ping-pong [A, B]
    sort_particles_data_bind_groups: [Arc<wgpu::BindGroup>; 2],
    sort_particles_params_bind_group: Arc<wgpu::BindGroup>,

    // Pipelines
    clear_bin_size_pipeline: Arc<wgpu::ComputePipeline>,
    fill_bin_size_pipeline: Arc<wgpu::ComputePipeline>,
    prefix_sum_pipeline: Arc<wgpu::ComputePipeline>,
    sort_particles_pipeline: Arc<wgpu::ComputePipeline>,
    sort_clear_bin_pipeline: Arc<wgpu::ComputePipeline>,

    num_bins: u32,
}

#[derive(Clone)]
struct Compute {
    // Dual particle buffers for ping-pong [A, B]
    particles_bufs: [Arc<wgpu::Buffer>; 2],
    sim_params_buf: Arc<wgpu::Buffer>,
    // Dual bind groups for ping-pong [A, B]
    particle_bind_groups: [Arc<wgpu::BindGroup>; 2],
    particle_pipeline: Arc<wgpu::ComputePipeline>,

    #[cfg(not(target_arch = "wasm32"))]
    audio_out_buf: Arc<wgpu::Buffer>,
    #[cfg(not(target_arch = "wasm32"))]
    audio_staging_bufs: Vec<Arc<wgpu::Buffer>>,
    #[cfg(not(target_arch = "wasm32"))]
    audio_params_buf: Arc<wgpu::Buffer>,
    #[cfg(not(target_arch = "wasm32"))]
    audio_bind_groups: [Arc<wgpu::BindGroup>; 2],
    #[cfg(not(target_arch = "wasm32"))]
    audio_pipeline: Arc<wgpu::ComputePipeline>,

    #[cfg(not(target_arch = "wasm32"))]
    phase_update_bind_groups: [Arc<wgpu::BindGroup>; 2],
    #[cfg(not(target_arch = "wasm32"))]
    phase_update_pipeline: Arc<wgpu::ComputePipeline>,

    spatial_hash: SpatialHash,
}

#[derive(Clone)]
struct Render {
    vertex_buffer: Arc<wgpu::Buffer>,
    index_buffer: Arc<wgpu::Buffer>,
    render_params_buf: Arc<wgpu::Buffer>,
    // Dual bind groups for ping-pong [A, B]
    bind_groups: [Arc<wgpu::BindGroup>; 2],
    pipeline: Arc<wgpu::RenderPipeline>,
    noise_texture: Arc<wgpu::TextureHandle>,
    noise_texture_view: Arc<wgpu::TextureViewHandle>,
    noise_sampler: Arc<wgpu::Sampler>,
}

#[derive(Clone)]
struct Trail {
    textures: [Arc<wgpu::TextureHandle>; 2],
    texture_views: [Arc<wgpu::TextureViewHandle>; 2],
    reproject_params_buf: Arc<wgpu::Buffer>,
    reproject_bind_groups: [Arc<wgpu::BindGroup>; 2],  // [i] reads texture[i]
    reproject_pipeline: Arc<wgpu::RenderPipeline>,
    blit_bind_groups: [Arc<wgpu::BindGroup>; 2],  // [i] reads texture[i]
    blit_pipeline: Arc<wgpu::RenderPipeline>,
    sampler: Arc<wgpu::Sampler>,
    ping_pong_idx: Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Clone)]
struct Model {
    compute: Compute,
    render: Render,
    trail: Trail,

    #[cfg(not(target_arch = "wasm32"))]
    audio_producer: Arc<Mutex<ringbuf::HeapProd<f32>>>,
    #[cfg(not(target_arch = "wasm32"))]
    audio_feedback_rx: Arc<Receiver<AudioFeedback>>,
    #[cfg(not(target_arch = "wasm32"))]
    _audio_thread: Arc<JoinHandle<()>>,

    window: Entity,
    settings: Settings,

    time: f32,
    frame_count: u64,

    map_x0: f32,
    map_x1: f32,
    map_y0: f32,
    map_y1: f32,

    current_x0: f32,
    current_x1: f32,
    current_y0: f32,
    current_y1: f32,

    prev_viewport: Arc<Mutex<[f32; 4]>>,

    #[cfg(not(target_arch = "wasm32"))]
    request_threshold: u32,
    #[cfg(not(target_arch = "wasm32"))]
    integral_error: f32,
    #[cfg(not(target_arch = "wasm32"))]
    latest_feedback: Option<AudioFeedback>,
    #[cfg(not(target_arch = "wasm32"))]
    last_buffer_level: u32,

    #[cfg(not(target_arch = "wasm32"))]
    last_log_frame: u64,

    #[cfg(not(target_arch = "wasm32"))]
    last_audio_staging_idx: Arc<AtomicI8>,

    current_buffer_idx: Arc<std::sync::atomic::AtomicUsize>,

    pan_vel_x: f32,
    pan_vel_y: f32,
    zoom_vel: f32,

    dragging: bool,
    last_mouse: [f32; 2],
}

#[cfg(not(target_arch = "wasm32"))]
struct AudioModel {
    consumer: ringbuf::HeapCons<f32>,
    feedback_tx: crossbeam_channel::Sender<AudioFeedback>,

    // Rolling buffer of recent buffer levels for min calculation
    buffer_history: VecDeque<u32>,
}

#[derive(Clone)]
struct Settings {
    volume: f32,
    dt: f32,
    friction: f32,
    mass: f32,
    radius: f32,
    collision_radius: f32,
    collision_strength: f32,
    max_force_strength: f32,
    copy_radius: f32,
    copy_cos_sim_threshold: f32,
    copy_probability: f32,
    trail_decay: f32,
}

fn main() {
    nannou::app(model).update(update).render(render).exit(exit).run();
}

fn model(app: &App) -> Model {
    // Create window
    #[cfg(target_arch = "wasm32")]
    let (init_w, init_h) = {
        let win = web_sys::window().unwrap();
        (
            win.inner_width().unwrap().as_f64().unwrap() as u32,
            win.inner_height().unwrap().as_f64().unwrap() as u32,
        )
    };
    #[cfg(not(target_arch = "wasm32"))]
    let (init_w, init_h) = (800, 800);

    let mut window_builder = app
        .new_window::<Model>()
        .hdr(true)
        .size(init_w, init_h)
        .key_pressed(key_pressed)
        .mouse_wheel(mouse_wheel)
        .mouse_pressed(mouse_pressed)
        .mouse_released(mouse_released)
        .mouse_moved(mouse_moved);
    #[cfg(target_arch = "wasm32")]
    {
        window_builder = window_builder.primary();
    }
    let w_id = window_builder.build();

    let window = app.window(w_id);
    let device = window.device();
    let queue = window.queue();

    #[cfg(not(target_arch = "wasm32"))]
    let (audio_producer, audio_feedback_rx, audio_thread) = {
        // Create ring buffer for lock-free audio transfer
        // Size: enough for ~32 update frames worth of audio
        let ring_buf_size = CHUNK_SIZE as usize * NUM_CHANNELS as usize * 32;
        let ring_buf = HeapRb::<f32>::new(ring_buf_size);
        let (audio_producer, audio_consumer) = ring_buf.split();

        // Start audio stream on a separate thread
        let audio_host = audio::Host::new();
        let (audio_feedback_tx, audio_feedback_rx) = crossbeam_channel::unbounded();

        let audio_model = AudioModel {
            consumer: audio_consumer,
            feedback_tx: audio_feedback_tx,
            buffer_history: VecDeque::with_capacity(BUFFER_HISTORY_SIZE),
        };

        let audio_thread = std::thread::spawn(move || {
            let stream = audio_host
                .new_output_stream(audio_model)
                .render(audio_fn)
                .channels(NUM_CHANNELS as usize)
                .sample_rate(SAMPLE_RATE)
                .build()
                .unwrap();
            stream.play().unwrap();

            // Keep thread alive - stream runs in background
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });

        (audio_producer, audio_feedback_rx, audio_thread)
    };

    let map_x0 = -8.0;
    let map_x1 = 8.0;
    let map_y0 = -8.0;
    let map_y1 = 8.0;

    // Adjust viewport to match window aspect ratio so particles render circular
    let win_size = window.size_pixels();
    let aspect = win_size.x as f32 / win_size.y as f32;
    let (current_x0, current_x1, current_y0, current_y1) = if aspect >= 1.0 {
        // Wider than tall: expand X range
        (map_x0 * aspect, map_x1 * aspect, map_y0, map_y1)
    } else {
        // Taller than wide: expand Y range
        (map_x0, map_x1, map_y0 / aspect, map_y1 / aspect)
    };

    // Initialize particles in spatial bins with per-bin species
    let min_bin_species = 1;
    let max_bin_species = 3;
    let initial_velocity = 0.1;

    // Calculate grid dimensions for initialization bins
    let bin_size = 2.0;
    let grid_size_x = ((map_x1 - map_x0) / bin_size).ceil() as u32;
    let grid_size_y = ((map_y1 - map_y0) / bin_size).ceil() as u32;
    let bin_count = grid_size_x * grid_size_y;

    // Species definition for a bin
    struct BinSpecies {
        sx: f32,
        sy: f32,
        ax: f32,
        ay: f32,
    }

    let mut particles = Vec::with_capacity(NUM_PARTICLES as usize);

    for j in 0..grid_size_y {
        for i in 0..grid_size_x {
            let bin_index = j * grid_size_x + i;
            let bin_start = (NUM_PARTICLES * bin_index / bin_count) as usize;
            let bin_end = if i == grid_size_x - 1 && j == grid_size_y - 1 {
                NUM_PARTICLES as usize
            } else {
                (NUM_PARTICLES * (bin_index + 1) / bin_count) as usize
            };
            let bin_particle_count = bin_end - bin_start;

            // Create random species set for this bin
            let num_bin_species = min_bin_species + (random_f32() * (max_bin_species - min_bin_species + 1) as f32) as usize;
            let num_bin_species = num_bin_species.min(max_bin_species);
            let species_in_bin: Vec<BinSpecies> = (0..num_bin_species)
                .map(|_| BinSpecies {
                    sx: random_f32() * 2.0 - 1.0,
                    sy: random_f32() * 2.0 - 1.0,
                    ax: random_f32() * 2.0 - 1.0,
                    ay: random_f32() * 2.0 - 1.0,
                })
                .collect();

            for k in 0..bin_particle_count {
                let species_id = k % num_bin_species;
                let species = &species_in_bin[species_id];

                // Position within this bin
                let x = map_x0 + (i as f32 + random_f32()) * (map_x1 - map_x0) / grid_size_x as f32;
                let y = map_y0 + (j as f32 + random_f32()) * (map_y1 - map_y0) / grid_size_y as f32;

                particles.push(Particle {
                    pos: [x, y],
                    vel: [
                        initial_velocity * (random_f32() * 2.0 - 1.0),
                        initial_velocity * (random_f32() * 2.0 - 1.0),
                    ],
                    phase: random_f32(),
                    energy: 0.0,
                    species: [species.sx, species.sy],
                    alpha: [species.ax, species.ay],
                    interaction: [0.0; 2],
                    amp_phase: random_f32(),
                    _pad: 0.0,
                });
            }
        }
    }

    let particles_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("particles"),
        contents: bytemuck::cast_slice(&particles),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim_params"),
        size: std::mem::size_of::<SimParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Particle compute pipeline
    let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle_shader"),
        source: wgpu::ShaderSource::Wgsl(PARTICLE_SHADER.into()),
    });

    let particles_buf_size = (NUM_PARTICLES as usize * std::mem::size_of::<Particle>()) as wgpu::BufferAddress;

    // Note: particle_bind_group_layout, particle_bind_group, and particle_pipeline
    // are created after spatial hash buffers since they depend on bin_offset_buf

    // Audio compute pipeline (native only)
    #[cfg(not(target_arch = "wasm32"))]
    let audio_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_out"),
        size: (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[cfg(not(target_arch = "wasm32"))]
    let audio_staging_bufs: Vec<Arc<wgpu::Buffer>> = (0..NUM_AUDIO_STAGING_BUFS)
        .map(|i| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("audio_staging_{}", i)),
                size: (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        })
        .collect();

    #[cfg(not(target_arch = "wasm32"))]
    let audio_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("audio_params"),
        size: std::mem::size_of::<AudioParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    #[cfg(not(target_arch = "wasm32"))]
    let audio_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("audio_shader"),
        source: wgpu::ShaderSource::Wgsl(AUDIO_SHADER.into()),
    });

    #[cfg(not(target_arch = "wasm32"))]
    let audio_out_size = (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;

    #[cfg(not(target_arch = "wasm32"))]
    let phase_update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("phase_update_shader"),
        source: wgpu::ShaderSource::Wgsl(PHASE_UPDATE_SHADER.into()),
    });

    #[cfg(not(target_arch = "wasm32"))]
    let phase_update_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // particles read-write
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)
        .build(&device);

    #[cfg(not(target_arch = "wasm32"))]
    let phase_update_bind_group_a = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<AudioParams>(&audio_params_buf, 0..1)
        .build(&device, &phase_update_bind_group_layout);

    #[cfg(not(target_arch = "wasm32"))]
    let phase_update_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("phase_update_pl"),
        bind_group_layouts: &[&phase_update_bind_group_layout],
        push_constant_ranges: &[],
    });

    #[cfg(not(target_arch = "wasm32"))]
    let phase_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("phase_update_pipeline"),
        layout: Some(&phase_update_pipeline_layout),
        module: &phase_update_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ==================== Spatial Hashing Setup ====================
    let num_bins_x = ((map_x1 - map_x0) / BIN_SIZE).ceil() as u32;
    let num_bins_y = ((map_y1 - map_y0) / BIN_SIZE).ceil() as u32;
    let num_bins = num_bins_x * num_bins_y;
    let bin_buf_size = ((num_bins + 1) * std::mem::size_of::<u32>() as u32) as u64;

    // Bin size buffer (atomic u32 array)
    let bin_size_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bin_size"),
        size: bin_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Bin offset buffer (prefix sum result)
    let bin_offset_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bin_offset"),
        size: bin_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Temp buffer for prefix sum ping-pong
    let bin_offset_tmp_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bin_offset_tmp"),
        size: bin_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Sorted particles buffer
    let particles_sorted_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles_sorted"),
        size: particles_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[cfg(not(target_arch = "wasm32"))]
    let phase_update_bind_group_b = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_sorted_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<AudioParams>(&audio_params_buf, 0..1)
        .build(&device, &phase_update_bind_group_layout);

    // Prefix sum step size uniform
    let prefix_sum_step_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prefix_sum_step"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load spatial hashing shaders
    let bin_fill_size_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bin_fill_size_shader"),
        source: wgpu::ShaderSource::Wgsl(BIN_FILL_SIZE_SHADER.into()),
    });

    let bin_prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bin_prefix_sum_shader"),
        source: wgpu::ShaderSource::Wgsl(BIN_PREFIX_SUM_SHADER.into()),
    });

    let particle_sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle_sort_shader"),
        source: wgpu::ShaderSource::Wgsl(PARTICLE_SORT_SHADER.into()),
    });

    // Fill bin size bind groups (particles, params, binSize)
    let fill_bin_size_particles_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true) // particles read-only
        .build(&device);

    let fill_bin_size_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false) // params
        .build(&device);

    let fill_bin_size_bins_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // binSize atomic
        .build(&device);

    // Dual bind groups for ping-pong: A reads from particles_buf, B reads from particles_sorted_buf
    let fill_bin_size_particles_bind_group_a = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .build(&device, &fill_bin_size_particles_layout);

    let fill_bin_size_particles_bind_group_b = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_sorted_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .build(&device, &fill_bin_size_particles_layout);

    let fill_bin_size_params_bind_group = wgpu::BindGroupBuilder::new()
        .buffer::<SimParams>(&sim_params_buf, 0..1)
        .build(&device, &fill_bin_size_params_layout);

    let fill_bin_size_bins_bind_group = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&bin_size_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .build(&device, &fill_bin_size_bins_layout);

    let fill_bin_size_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fill_bin_size_pl"),
        bind_group_layouts: &[&fill_bin_size_particles_layout, &fill_bin_size_params_layout, &fill_bin_size_bins_layout],
        push_constant_ranges: &[],
    });

    // Clear bin size pipeline (uses same layout as fill - shader declares all 3 groups)
    let clear_bin_size_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("clear_bin_size_pipeline"),
        layout: Some(&fill_bin_size_pipeline_layout),
        module: &bin_fill_size_shader,
        entry_point: Some("clearBinSize"),
        compilation_options: Default::default(),
        cache: None,
    });

    let fill_bin_size_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fill_bin_size_pipeline"),
        layout: Some(&fill_bin_size_pipeline_layout),
        module: &bin_fill_size_shader,
        entry_point: Some("fillBinSize"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Prefix sum bind groups (ping-pong between two buffers)
    let prefix_sum_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true)  // source (read-only)
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // destination
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)        // stepSize
        .build(&device);

    // Bind group 0: bin_size -> bin_offset (first iteration)
    let prefix_sum_bind_group_0 = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&bin_size_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .buffer::<u32>(&prefix_sum_step_buf, 0..1)
        .build(&device, &prefix_sum_bind_group_layout);

    // Bind group 1: bin_offset -> bin_offset_tmp
    let prefix_sum_bind_group_1 = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .buffer_bytes(&bin_offset_tmp_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .buffer::<u32>(&prefix_sum_step_buf, 0..1)
        .build(&device, &prefix_sum_bind_group_layout);

    // Bind group 2: bin_offset_tmp -> bin_offset
    let prefix_sum_bind_group_2 = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&bin_offset_tmp_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .buffer::<u32>(&prefix_sum_step_buf, 0..1)
        .build(&device, &prefix_sum_bind_group_layout);

    let prefix_sum_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("prefix_sum_pl"),
        bind_group_layouts: &[&prefix_sum_bind_group_layout],
        push_constant_ranges: &[],
    });

    let prefix_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("prefix_sum_pipeline"),
        layout: Some(&prefix_sum_pipeline_layout),
        module: &bin_prefix_sum_shader,
        entry_point: Some("prefixSumStep"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Sort particles bind groups
    let sort_particles_data_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true)  // source particles
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // destination particles
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true)  // binOffset
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // binSize (atomic)
        .build(&device);

    let sort_particles_params_layout = wgpu::BindGroupLayoutBuilder::new()
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false) // params
        .build(&device);

    // Dual bind groups for ping-pong sorting:
    // A: particles_buf -> particles_sorted_buf
    // B: particles_sorted_buf -> particles_buf
    let sort_particles_data_bind_group_a = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer_bytes(&particles_sorted_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .buffer_bytes(&bin_size_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .build(&device, &sort_particles_data_layout);

    let sort_particles_data_bind_group_b = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_sorted_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .buffer_bytes(&bin_size_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .build(&device, &sort_particles_data_layout);

    let sort_particles_params_bind_group = wgpu::BindGroupBuilder::new()
        .buffer::<SimParams>(&sim_params_buf, 0..1)
        .build(&device, &sort_particles_params_layout);

    let sort_particles_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sort_particles_pl"),
        bind_group_layouts: &[&sort_particles_data_layout, &sort_particles_params_layout],
        push_constant_ranges: &[],
    });

    let sort_particles_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sort_particles_pipeline"),
        layout: Some(&sort_particles_pipeline_layout),
        module: &particle_sort_shader,
        entry_point: Some("sortParticles"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Sort clear bin pipeline (reuses the data bind group)
    let sort_clear_bin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sort_clear_bin_pipeline"),
        layout: Some(&sort_particles_pipeline_layout),
        module: &particle_sort_shader,
        entry_point: Some("clearBinSize"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Now create particle bind groups for ping-pong buffers
    // Each frame: sort from buffer A -> B, then simulate in B (or vice versa)
    let particle_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // particles read-write
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)        // params
        .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true)  // bin_offset read-only
        .build(&device);

    // Bind group 0: operates on particles_buf (buffer A)
    let particle_bind_group_a = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<SimParams>(&sim_params_buf, 0..1)
        .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .build(&device, &particle_bind_group_layout);

    // Bind group 1: operates on particles_sorted_buf (buffer B)
    let particle_bind_group_b = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&particles_sorted_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<SimParams>(&sim_params_buf, 0..1)
        .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
        .build(&device, &particle_bind_group_layout);

    let particle_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle_pl"),
        bind_group_layouts: &[&particle_bind_group_layout],
        push_constant_ranges: &[],
    });

    let particle_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("particle_pipeline"),
        layout: Some(&particle_pipeline_layout),
        module: &particle_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Audio bind groups and pipeline (native only)
    #[cfg(not(target_arch = "wasm32"))]
    let (audio_bind_group_a, audio_bind_group_b, audio_pipeline) = {
        let audio_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
            .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true)  // particles read-only
            .storage_buffer(wgpu::ShaderStages::COMPUTE, false, false) // audio_out
            .uniform_buffer(wgpu::ShaderStages::COMPUTE, false)        // params
            .storage_buffer(wgpu::ShaderStages::COMPUTE, false, true)  // bin_offset read-only
            .build(&device);

        let audio_bind_group_a = wgpu::BindGroupBuilder::new()
            .buffer_bytes(&particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
            .buffer_bytes(&audio_out_buf, 0, std::num::NonZeroU64::new(audio_out_size))
            .buffer::<AudioParams>(&audio_params_buf, 0..1)
            .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
            .build(&device, &audio_bind_group_layout);

        let audio_bind_group_b = wgpu::BindGroupBuilder::new()
            .buffer_bytes(&particles_sorted_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
            .buffer_bytes(&audio_out_buf, 0, std::num::NonZeroU64::new(audio_out_size))
            .buffer::<AudioParams>(&audio_params_buf, 0..1)
            .buffer_bytes(&bin_offset_buf, 0, std::num::NonZeroU64::new(bin_buf_size))
            .build(&device, &audio_bind_group_layout);

        let audio_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("audio_pl"),
            bind_group_layouts: &[&audio_bind_group_layout],
            push_constant_ranges: &[],
        });

        let audio_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("audio_pipeline"),
            layout: Some(&audio_pipeline_layout),
            module: &audio_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Write initial audio params
        queue.write_buffer(
            &audio_params_buf,
            0,
            bytemuck::bytes_of(&AudioParams {
                sample_rate: SAMPLE_RATE as f32,
                num_particles: NUM_PARTICLES,
                chunk_size: CHUNK_SIZE,
                volume: 0.8,
                current_x: [current_x0, current_x1],
                current_y: [current_y0, current_y1],
                map_x0,
                map_y0,
                bin_size: BIN_SIZE,
                num_bins_x,
                num_bins_y,
                max_speed: 4.0,
                energy_scale: 4.0,
                _pad: 0,
            }),
        );

        (audio_bind_group_a, audio_bind_group_b, audio_pipeline)
    };

    let spatial_hash = SpatialHash {
        bin_size_buf: Arc::new(bin_size_buf),
        bin_offset_buf: Arc::new(bin_offset_buf),
        bin_offset_tmp_buf: Arc::new(bin_offset_tmp_buf),
        prefix_sum_step_buf: Arc::new(prefix_sum_step_buf),
        fill_bin_size_particles_bind_groups: [Arc::new(fill_bin_size_particles_bind_group_a), Arc::new(fill_bin_size_particles_bind_group_b)],
        fill_bin_size_params_bind_group: Arc::new(fill_bin_size_params_bind_group),
        fill_bin_size_bins_bind_group: Arc::new(fill_bin_size_bins_bind_group),
        prefix_sum_bind_groups: [Arc::new(prefix_sum_bind_group_0), Arc::new(prefix_sum_bind_group_1), Arc::new(prefix_sum_bind_group_2)],
        sort_particles_data_bind_groups: [Arc::new(sort_particles_data_bind_group_a), Arc::new(sort_particles_data_bind_group_b)],
        sort_particles_params_bind_group: Arc::new(sort_particles_params_bind_group),
        clear_bin_size_pipeline: Arc::new(clear_bin_size_pipeline),
        fill_bin_size_pipeline: Arc::new(fill_bin_size_pipeline),
        prefix_sum_pipeline: Arc::new(prefix_sum_pipeline),
        sort_particles_pipeline: Arc::new(sort_particles_pipeline),
        sort_clear_bin_pipeline: Arc::new(sort_clear_bin_pipeline),
        num_bins,
    };

    let particles_buf = Arc::new(particles_buf);
    let particles_sorted_buf = Arc::new(particles_sorted_buf);
    let compute = Compute {
        particles_bufs: [particles_buf.clone(), particles_sorted_buf.clone()],
        sim_params_buf: Arc::new(sim_params_buf),
        particle_bind_groups: [Arc::new(particle_bind_group_a), Arc::new(particle_bind_group_b)],
        particle_pipeline: Arc::new(particle_pipeline),
        #[cfg(not(target_arch = "wasm32"))]
        audio_out_buf: Arc::new(audio_out_buf),
        #[cfg(not(target_arch = "wasm32"))]
        audio_staging_bufs,
        #[cfg(not(target_arch = "wasm32"))]
        audio_params_buf: Arc::new(audio_params_buf),
        #[cfg(not(target_arch = "wasm32"))]
        audio_bind_groups: [Arc::new(audio_bind_group_a), Arc::new(audio_bind_group_b)],
        #[cfg(not(target_arch = "wasm32"))]
        audio_pipeline: Arc::new(audio_pipeline),
        #[cfg(not(target_arch = "wasm32"))]
        phase_update_bind_groups: [Arc::new(phase_update_bind_group_a), Arc::new(phase_update_bind_group_b)],
        #[cfg(not(target_arch = "wasm32"))]
        phase_update_pipeline: Arc::new(phase_update_pipeline),
        spatial_hash,
    };

    // Load blue noise texture from embedded PNG
    let noise_png_bytes = include_bytes!("blue_noise.png");
    let noise_img = ImageReader::new(std::io::Cursor::new(noise_png_bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
    let (noise_w, noise_h) = noise_img.dimensions();
    let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("noise_texture"),
        size: wgpu::Extent3d {
            width: noise_w,
            height: noise_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &noise_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &noise_img,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * noise_w),
            rows_per_image: Some(noise_h),
        },
        wgpu::Extent3d {
            width: noise_w,
            height: noise_h,
            depth_or_array_layers: 1,
        },
    );
    let noise_texture_view = noise_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("noise_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        ..Default::default()
    });

    // Create render pipeline for GPU-based particle rendering with instancing
    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render_shader"),
        source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
    });

    // Vertex buffer for the quad
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("quad_vertices"),
        contents: bytemuck::cast_slice(&QUAD_VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Index buffer for the quad
    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("quad_indices"),
        contents: bytemuck::cast_slice(&QUAD_INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Render params uniform buffer
    let render_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_params"),
        size: std::mem::size_of::<RenderParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Render bind group layout: particles storage + params uniform + noise texture + sampler
    let render_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(wgpu::ShaderStages::VERTEX, false, true) // particles read-only
        .uniform_buffer(wgpu::ShaderStages::VERTEX_FRAGMENT, false) // render params (fragment needs max_speed)
        .texture(wgpu::ShaderStages::FRAGMENT, false, wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Float { filterable: true }) // noise texture
        .sampler(wgpu::ShaderStages::FRAGMENT, true) // noise sampler
        .build(&device);

    // Dual bind groups for ping-pong
    let render_bind_group_a = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&*particles_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<RenderParams>(&render_params_buf, 0..1)
        .texture_view(&noise_texture_view)
        .sampler(&noise_sampler)
        .build(&device, &render_bind_group_layout);

    let render_bind_group_b = wgpu::BindGroupBuilder::new()
        .buffer_bytes(&*particles_sorted_buf, 0, std::num::NonZeroU64::new(particles_buf_size))
        .buffer::<RenderParams>(&render_params_buf, 0..1)
        .texture_view(&noise_texture_view)
        .sampler(&noise_sampler)
        .build(&device, &render_bind_group_layout);

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipeline_layout"),
        bind_group_layouts: &[&render_bind_group_layout],
        push_constant_ranges: &[],
    });

    let format = Frame::TEXTURE_FORMAT;
    let msaa_samples = window.msaa_samples();

    let render_pipeline = wgpu::RenderPipelineBuilder::from_layout(&render_pipeline_layout, &render_shader)
        .vertex_entry_point("vs_main")
        .fragment_shader(&render_shader)
        .fragment_entry_point("fs_main")
        .color_format(format)
        .color_blend(wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::One,  // Additive blending - order independent
            operation: wgpu::BlendOperation::Add,
        })
        .alpha_blend(wgpu::BlendComponent::OVER)
        .add_vertex_buffer::<QuadVertex>(&QUAD_VERTEX_ATTRS)
        .sample_count(1)  // No MSAA - rendering to trail texture
        .primitive_topology(wgpu::PrimitiveTopology::TriangleList)
        .build(&*device);

    // ==================== Trail (Ping-Pong Reprojection) Setup ====================
    let window_size = window.size_pixels();

    // Create two trail textures for ping-pong reprojection
    let trail_texture_0 = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trail_texture_0"),
        size: wgpu::Extent3d {
            width: window_size.x,
            height: window_size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let trail_texture_view_0 = trail_texture_0.create_view(&wgpu::TextureViewDescriptor::default());

    let trail_texture_1 = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trail_texture_1"),
        size: wgpu::Extent3d {
            width: window_size.x,
            height: window_size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let trail_texture_view_1 = trail_texture_1.create_view(&wgpu::TextureViewDescriptor::default());

    let fullscreen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fullscreen_shader"),
        source: wgpu::ShaderSource::Wgsl(FULLSCREEN_SHADER.into()),
    });

    let trail_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("trail_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // Reproject pipeline (samples previous trail texture with viewport reprojection + decay)
    let reproject_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reproject_params"),
        size: std::mem::size_of::<ReprojectParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let reproject_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .texture(wgpu::ShaderStages::FRAGMENT, false, wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Float { filterable: true })
        .sampler(wgpu::ShaderStages::FRAGMENT, true)
        .uniform_buffer(wgpu::ShaderStages::FRAGMENT, false)
        .build(&device);

    // Bind group [i] reads from texture[i]
    let reproject_bind_group_0 = wgpu::BindGroupBuilder::new()
        .texture_view(&trail_texture_view_0)
        .sampler(&trail_sampler)
        .buffer::<ReprojectParams>(&reproject_params_buf, 0..1)
        .build(&device, &reproject_bind_group_layout);

    let reproject_bind_group_1 = wgpu::BindGroupBuilder::new()
        .texture_view(&trail_texture_view_1)
        .sampler(&trail_sampler)
        .buffer::<ReprojectParams>(&reproject_params_buf, 0..1)
        .build(&device, &reproject_bind_group_layout);

    let reproject_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("reproject_pipeline_layout"),
        bind_group_layouts: &[&reproject_bind_group_layout],
        push_constant_ranges: &[],
    });

    let reproject_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("reproject_pipeline"),
        layout: Some(&reproject_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &fullscreen_shader,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fullscreen_shader,
            entry_point: Some("fs_reproject"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,  // Direct write - shader computes final color with decay
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // Blit pipeline (simple passthrough - trail texture is already in viewport space)
    let blit_bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .texture(wgpu::ShaderStages::FRAGMENT, false, wgpu::TextureViewDimension::D2, wgpu::TextureSampleType::Float { filterable: true })
        .sampler(wgpu::ShaderStages::FRAGMENT, true)
        .build(&device);

    let blit_bind_group_0 = wgpu::BindGroupBuilder::new()
        .texture_view(&trail_texture_view_0)
        .sampler(&trail_sampler)
        .build(&device, &blit_bind_group_layout);

    let blit_bind_group_1 = wgpu::BindGroupBuilder::new()
        .texture_view(&trail_texture_view_1)
        .sampler(&trail_sampler)
        .build(&device, &blit_bind_group_layout);

    let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit_pipeline_layout"),
        bind_group_layouts: &[&blit_bind_group_layout],
        push_constant_ranges: &[],
    });

    let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit_pipeline"),
        layout: Some(&blit_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &fullscreen_shader,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fullscreen_shader,
            entry_point: Some("fs_blit"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: msaa_samples,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    });

    let trail = Trail {
        textures: [Arc::new(trail_texture_0), Arc::new(trail_texture_1)],
        texture_views: [Arc::new(trail_texture_view_0), Arc::new(trail_texture_view_1)],
        reproject_params_buf: Arc::new(reproject_params_buf),
        reproject_bind_groups: [Arc::new(reproject_bind_group_0), Arc::new(reproject_bind_group_1)],
        reproject_pipeline: Arc::new(reproject_pipeline),
        blit_bind_groups: [Arc::new(blit_bind_group_0), Arc::new(blit_bind_group_1)],
        blit_pipeline: Arc::new(blit_pipeline),
        sampler: Arc::new(trail_sampler),
        ping_pong_idx: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };

    let render = Render {
        vertex_buffer: Arc::new(vertex_buffer),
        index_buffer: Arc::new(index_buffer),
        render_params_buf: Arc::new(render_params_buf),
        bind_groups: [Arc::new(render_bind_group_a), Arc::new(render_bind_group_b)],
        pipeline: Arc::new(render_pipeline),
        noise_texture: Arc::new(noise_texture),
        noise_texture_view: Arc::new(noise_texture_view),
        noise_sampler: Arc::new(noise_sampler),
    };

    Model {
        compute,
        render,
        trail,
        #[cfg(not(target_arch = "wasm32"))]
        audio_producer: Arc::new(Mutex::new(audio_producer)),
        #[cfg(not(target_arch = "wasm32"))]
        audio_feedback_rx: Arc::new(audio_feedback_rx),
        #[cfg(not(target_arch = "wasm32"))]
        _audio_thread: Arc::new(audio_thread),
        window: w_id,
        settings: Settings {
            volume: 1.0,
            dt: 0.035,
            friction: 0.1,
            mass: 1.0,
            radius: 0.3,
            collision_radius: 0.1,
            collision_strength: 15.0,
            max_force_strength: 1.0,
            copy_radius: 0.2,
            copy_cos_sim_threshold: 0.2,
            copy_probability: 0.001,
            trail_decay: 0.6,
        },
        time: 0.0,
        frame_count: 0,
        map_x0,
        map_x1,
        map_y0,
        map_y1,
        current_x0,
        current_x1,
        current_y0,
        current_y1,
        prev_viewport: Arc::new(Mutex::new([current_x0, current_x1, current_y0, current_y1])),
        #[cfg(not(target_arch = "wasm32"))]
        request_threshold: INITIAL_REQUEST_THRESHOLD,
        #[cfg(not(target_arch = "wasm32"))]
        integral_error: 0.0,
        #[cfg(not(target_arch = "wasm32"))]
        latest_feedback: None,
        #[cfg(not(target_arch = "wasm32"))]
        last_buffer_level: 0,
        #[cfg(not(target_arch = "wasm32"))]
        last_log_frame: 0,
        #[cfg(not(target_arch = "wasm32"))]
        last_audio_staging_idx: Arc::new(AtomicI8::new(-1)),
        current_buffer_idx: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        pan_vel_x: 0.0,
        pan_vel_y: 0.0,
        zoom_vel: 0.0,
        dragging: false,
        last_mouse: [0.0, 0.0],
    }
}

fn exit(_app: &App, _model: Model) {
    // Audio thread will be cleaned up when Model is dropped
}

fn key_pressed(app: &App, model: &mut Model, key: KeyCode) {
    match key {
        KeyCode::KeyR => {
            let win_size = app.window(model.window).size_pixels();
            let aspect = win_size.x as f32 / win_size.y as f32;
            if aspect >= 1.0 {
                model.current_x0 = model.map_x0 * aspect;
                model.current_x1 = model.map_x1 * aspect;
                model.current_y0 = model.map_y0;
                model.current_y1 = model.map_y1;
            } else {
                model.current_x0 = model.map_x0;
                model.current_x1 = model.map_x1;
                model.current_y0 = model.map_y0 / aspect;
                model.current_y1 = model.map_y1 / aspect;
            }
            model.pan_vel_x = 0.0;
            model.pan_vel_y = 0.0;
            model.zoom_vel = 0.0;
        }
        _ => {}
    }
}

fn mouse_wheel(app: &App, model: &mut Model, wheel: MouseWheel) {
    let zoom_factor = 0.02;
    // Browsers report pixel deltas (e.g. 100.0) instead of line deltas (e.g. 1.0).
    // Use the sign to normalize so zoom works consistently across native and WASM.
    let y = wheel.y.signum() * wheel.y.abs().min(3.0);
    let scale = 1.0 - y * zoom_factor;

    // Map mouse screen position to world coordinates for zoom-to-cursor
    let mouse = app.mouse();
    let win_size = app.window(model.window).size_pixels();
    let norm_x = mouse.x / win_size.x as f32 + 0.5;
    let norm_y = mouse.y / win_size.y as f32 + 0.5;
    let world_x = model.current_x0 + norm_x * (model.current_x1 - model.current_x0);
    let world_y = model.current_y0 + norm_y * (model.current_y1 - model.current_y0);

    // Scale the viewport around the cursor's world position
    model.current_x0 = world_x + (model.current_x0 - world_x) * scale;
    model.current_x1 = world_x + (model.current_x1 - world_x) * scale;
    model.current_y0 = world_y + (model.current_y0 - world_y) * scale;
    model.current_y1 = world_y + (model.current_y1 - world_y) * scale;
}

fn mouse_pressed(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left {
        model.dragging = true;
    }
}

fn mouse_released(_app: &App, model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left {
        model.dragging = false;
    }
}

fn mouse_moved(_app: &App, model: &mut Model, pos: Point2) {
    if model.dragging {
        let win_size = _app.window(model.window).size_pixels();
        let dx = pos.x - model.last_mouse[0];
        let dy = pos.y - model.last_mouse[1];
        // Convert pixel delta to world delta
        let world_dx = dx / win_size.x as f32 * (model.current_x1 - model.current_x0);
        let world_dy = dy / win_size.y as f32 * (model.current_y1 - model.current_y0);
        model.current_x0 -= world_dx;
        model.current_x1 -= world_dx;
        model.current_y0 += world_dy;
        model.current_y1 += world_dy;
    }
    // Always track mouse position so last_mouse is current when a drag starts
    model.last_mouse = [pos.x, pos.y];
}

fn update(app: &App, model: &mut Model) {
    // Smooth viewport panning and zooming
    let keys = app.keys();
    if keys.pressed(KeyCode::KeyW) {
        model.pan_vel_y = (model.pan_vel_y + PAN_ACCEL).min(PAN_MAX_SPEED);
    }
    if keys.pressed(KeyCode::KeyS) {
        model.pan_vel_y = (model.pan_vel_y - PAN_ACCEL).max(-PAN_MAX_SPEED);
    }
    if keys.pressed(KeyCode::KeyA) {
        model.pan_vel_x = (model.pan_vel_x - PAN_ACCEL).max(-PAN_MAX_SPEED);
    }
    if keys.pressed(KeyCode::KeyD) {
        model.pan_vel_x = (model.pan_vel_x + PAN_ACCEL).min(PAN_MAX_SPEED);
    }
    if keys.pressed(KeyCode::KeyE) {
        model.zoom_vel = (model.zoom_vel - ZOOM_ACCEL).max(-ZOOM_MAX_SPEED);
    }
    if keys.pressed(KeyCode::KeyF) {
        model.zoom_vel = (model.zoom_vel + ZOOM_ACCEL).min(ZOOM_MAX_SPEED);
    }

    // Apply friction
    if !keys.pressed(KeyCode::KeyW) && !keys.pressed(KeyCode::KeyS) {
        model.pan_vel_y *= PAN_FRICTION;
    }
    if !keys.pressed(KeyCode::KeyA) && !keys.pressed(KeyCode::KeyD) {
        model.pan_vel_x *= PAN_FRICTION;
    }
    if !keys.pressed(KeyCode::KeyE) && !keys.pressed(KeyCode::KeyF) {
        model.zoom_vel *= ZOOM_FRICTION;
    }

    // Kill tiny velocities to avoid drift
    if model.pan_vel_x.abs() < 0.0001 { model.pan_vel_x = 0.0; }
    if model.pan_vel_y.abs() < 0.0001 { model.pan_vel_y = 0.0; }
    if model.zoom_vel.abs() < 0.0001 { model.zoom_vel = 0.0; }

    // Apply pan velocity (scaled by current view size for consistent feel)
    let view_width = model.current_x1 - model.current_x0;
    let view_height = model.current_y1 - model.current_y0;
    let dx = model.pan_vel_x * view_width;
    let dy = model.pan_vel_y * view_height;
    model.current_x0 += dx;
    model.current_x1 += dx;
    model.current_y0 += dy;
    model.current_y1 += dy;

    // Apply zoom velocity (zoom toward center)
    if model.zoom_vel != 0.0 {
        let center_x = (model.current_x0 + model.current_x1) / 2.0;
        let center_y = (model.current_y0 + model.current_y1) / 2.0;
        let scale = 1.0 + model.zoom_vel;
        let new_width = view_width * scale;
        let new_height = view_height * scale;
        model.current_x0 = center_x - new_width / 2.0;
        model.current_x1 = center_x + new_width / 2.0;
        model.current_y0 = center_y - new_height / 2.0;
        model.current_y1 = center_y + new_height / 2.0;
    }

    // egui UI for settings
    let mut egui_ctx = app.egui_for_window(model.window);
    let ctx = egui_ctx.get_mut();

    egui::Window::new("Settings").show(&ctx, |ui| {
        ui.add(egui::Slider::new(&mut model.settings.volume, 0.0..=2.0).text("volume"));
        ui.add(egui::Slider::new(&mut model.settings.dt, 0.001..=0.1).text("dt"));
        ui.add(egui::Slider::new(&mut model.settings.friction, 0.0..=1.0).text("friction"));
        ui.add(egui::Slider::new(&mut model.settings.mass, 0.1..=10.0).text("mass"));
        ui.add(egui::Slider::new(&mut model.settings.radius, 0.01..=1.0).text("radius"));
        ui.add(egui::Slider::new(&mut model.settings.collision_radius, 0.01..=0.5).text("collision radius"));
        ui.add(egui::Slider::new(&mut model.settings.collision_strength, 0.0..=20.0).text("collision strength"));
        ui.add(egui::Slider::new(&mut model.settings.max_force_strength, 0.0..=10.0).text("max force strength"));
        ui.add(egui::Slider::new(&mut model.settings.copy_radius, 0.01..=0.5).text("copy radius"));
        ui.add(egui::Slider::new(&mut model.settings.copy_cos_sim_threshold, 0.0..=1.0).text("copy cos sim threshold"));
        ui.add(egui::Slider::new(&mut model.settings.copy_probability, 0.0..=0.1).text("copy probability"));
        ui.add(egui::Slider::new(&mut model.settings.trail_decay, 0.5..=1.0).text("trail decay"));
    });

    // Process audio feedback (native only)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut latest_feedback = None;
        let mut had_underrun = false;
        while let Ok(feedback) = model.audio_feedback_rx.try_recv() {
            if feedback.current_buffer_level == 0 {
                had_underrun = true;
            }
            latest_feedback = Some(feedback);
        }

        if had_underrun {
            eprintln!("[AUDIO] Buffer underrun detected!");
        }

        if let Some(feedback) = &latest_feedback {
            let min_buffer = feedback.min_buffer_level as f32;
            let target = (CHUNK_FLOATS) as f32;

            let error = target - min_buffer;
            let kp = 0.01;
            let ki = 0.002;

            model.integral_error = (model.integral_error + error).clamp(-5000.0, 5000.0);
            let adjustment = kp * error + ki * model.integral_error;

            let new_threshold = ((model.request_threshold as f32) + adjustment) as u32;
            model.request_threshold = new_threshold.clamp(MIN_REQUEST_THRESHOLD, MAX_REQUEST_THRESHOLD);
        }

        if model.frame_count - model.last_log_frame >= AUDIO_LOG_INTERVAL_FRAMES {
            if let Some(feedback) = &latest_feedback {
                eprintln!(
                    "[AUDIO] Buffer: current={}, min={}, threshold={}",
                    feedback.current_buffer_level,
                    feedback.min_buffer_level,
                    model.request_threshold
                );
            }
            model.last_log_frame = model.frame_count;
        }

        if let Some(feedback) = latest_feedback {
            model.last_buffer_level = feedback.current_buffer_level;
            model.latest_feedback = Some(feedback);
        }
    }

    let dt = 1.0 / 60.0;
    model.time += dt;
    model.frame_count += 1;
}

#[cfg(not(target_arch = "wasm32"))]
fn audio_fn(audio: &mut AudioModel, buffer: &mut Buffer) {
    for frame in buffer.frames_mut() {
        let left = audio.consumer.try_pop().unwrap_or(0.0);
        let right = audio.consumer.try_pop().unwrap_or(0.0);
        frame[0] = left;
        frame[1] = right;
    }

    let buffer_len = audio.consumer.occupied_len() as u32;

    audio.buffer_history.push_back(buffer_len);
    if audio.buffer_history.len() > BUFFER_HISTORY_SIZE {
        audio.buffer_history.pop_front();
    }

    let min_buffer = audio.buffer_history.iter().copied().min().unwrap_or(0);

    audio.feedback_tx.send(AudioFeedback {
        min_buffer_level: min_buffer,
        current_buffer_level: buffer_len,
    }).ok();
}

/// Helper to update a GPU buffer via staging buffer copy.
/// Returns the staging buffer which must be kept alive until the encoder is submitted.
fn update_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    dest: &wgpu::Buffer,
    data: &T,
) {
    let bytes = bytemuck::bytes_of(data);
    let staging = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("staging"),
        contents: bytes,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&staging, 0, dest, 0, bytes.len() as u64);
}

fn render(_app: &RenderApp, model: &Model, frame: Frame) {
    let device = frame.device();

    // Ping-pong buffer indices:
    // src_idx = current buffer (where data is now)
    // dst_idx = destination buffer (where sorted data will go, and where simulation runs)
    let src_idx = model.current_buffer_idx.load(Ordering::Relaxed);
    let dst_idx = 1 - src_idx;

    let mut encoder = frame.command_encoder();

    // Compute normalization parameters from simulation settings
    // Assumes ~20 average neighbors within interaction radius
    let avg_neighbors = 20.0_f32;
    // Terminal velocity: v ≈ F * friction / mass, where F ≈ avg_neighbors * 2 * max_force
    let max_speed = avg_neighbors * 2.0 * model.settings.max_force_strength * model.settings.friction / model.settings.mass;
    // Energy per neighbor ≈ max_force * (radius - collision_radius), summed over neighbors
    let energy_scale = avg_neighbors * model.settings.max_force_strength * (model.settings.radius - model.settings.collision_radius);

    // Update render params with current viewport
    let render_params = RenderParams {
        map_x: [model.map_x0, model.map_x1],
        map_y: [model.map_y0, model.map_y1],
        current_x: [model.current_x0, model.current_x1],
        current_y: [model.current_y0, model.current_y1],
        particle_size: PARTICLE_SIZE,
        num_particles: NUM_PARTICLES,
        max_speed,
        energy_scale,
    };
    update_buffer(device, &mut encoder, &*model.render.render_params_buf, &render_params);

    // Calculate bin counts
    let num_bins_x = ((model.map_x1 - model.map_x0) / BIN_SIZE).ceil() as u32;
    let num_bins_y = ((model.map_y1 - model.map_y0) / BIN_SIZE).ceil() as u32;

    // Update sim params
    let sim_params = SimParams {
        dt: model.settings.dt,
        time: model.time,
        num_particles: NUM_PARTICLES,
        friction: model.settings.friction,
        mass: model.settings.mass,
        map_x0: model.map_x0,
        map_x1: model.map_x1,
        map_y0: model.map_y0,
        map_y1: model.map_y1,
        bin_size: BIN_SIZE,
        num_bins_x,
        num_bins_y,
        radius: model.settings.radius,
        collision_radius: model.settings.collision_radius,
        collision_strength: model.settings.collision_strength,
        max_force_strength: model.settings.max_force_strength,
        copy_radius: model.settings.copy_radius,
        copy_cos_sim_threshold: model.settings.copy_cos_sim_threshold,
        copy_probability: model.settings.copy_probability,
        _pad: 0.0
    };
    update_buffer(device, &mut encoder, &*model.compute.sim_params_buf, &sim_params);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let audio_params = AudioParams {
            sample_rate: SAMPLE_RATE as f32,
            num_particles: NUM_PARTICLES,
            chunk_size: CHUNK_SIZE,
            volume: model.settings.volume,
            current_x: [model.current_x0, model.current_x1],
            current_y: [model.current_y0, model.current_y1],
            map_x0: model.map_x0,
            map_y0: model.map_y0,
            bin_size: BIN_SIZE,
            num_bins_x,
            num_bins_y,
            max_speed,
            energy_scale,
            _pad: 0,
        };
        update_buffer(device, &mut encoder, &*model.compute.audio_params_buf, &audio_params);
    }

    // Update reproject params for trail reprojection
    let prev = model.prev_viewport.lock().unwrap();
    let [prev_x0, prev_x1, prev_y0, prev_y1] = *prev;
    drop(prev);

    let reproject_params = ReprojectParams {
        prev_x: [prev_x0, prev_x1],
        prev_y: [prev_y0, prev_y1],
        current_x: [model.current_x0, model.current_x1],
        current_y: [model.current_y0, model.current_y1],
        decay: model.settings.trail_decay,
        _pad1: 0.0,
        _pad2: 0.0,
        _pad3: 0.0,
    };
    update_buffer(device, &mut encoder, &*model.trail.reproject_params_buf, &reproject_params);

    #[cfg(not(target_arch = "wasm32"))]
    let audio_buf_size = (CHUNK_SIZE * NUM_CHANNELS * std::mem::size_of::<f32>() as u32) as u64;

    // ==================== Spatial Hashing Passes ====================
    // Sort particles from src_idx buffer to dst_idx buffer
    let sh = &model.compute.spatial_hash;
    let num_bins_total = sh.num_bins + 1;

    // Step 1: Clear bin sizes
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clear_bin_size_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&*sh.clear_bin_size_pipeline);
        pass.set_bind_group(0, &*sh.fill_bin_size_particles_bind_groups[src_idx], &[]);
        pass.set_bind_group(1, &*sh.fill_bin_size_params_bind_group, &[]);
        pass.set_bind_group(2, &*sh.fill_bin_size_bins_bind_group, &[]);
        pass.dispatch_workgroups((num_bins_total + 63) / 64, 1, 1);
    }

    // Step 2: Fill bin sizes (from source buffer)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fill_bin_size_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&*sh.fill_bin_size_pipeline);
        pass.set_bind_group(0, &*sh.fill_bin_size_particles_bind_groups[src_idx], &[]);
        pass.set_bind_group(1, &*sh.fill_bin_size_params_bind_group, &[]);
        pass.set_bind_group(2, &*sh.fill_bin_size_bins_bind_group, &[]);
        pass.dispatch_workgroups((NUM_PARTICLES + 63) / 64, 1, 1);
    }

    // Step 3: Prefix sum (compute bin offsets)
    let num_prefix_sum_steps = (num_bins_total as f32).log2().ceil() as u32;

    for i in 0..num_prefix_sum_steps {
        let step_size = 1u32 << i;
        update_buffer(device, &mut encoder, &*sh.prefix_sum_step_buf, &step_size);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&*sh.prefix_sum_pipeline);

            let bind_group_idx = if i == 0 {
                0  // bin_size -> bin_offset
            } else if i % 2 == 1 {
                1  // bin_offset -> bin_offset_tmp
            } else {
                2  // bin_offset_tmp -> bin_offset
            };
            pass.set_bind_group(0, &*sh.prefix_sum_bind_groups[bind_group_idx], &[]);
            pass.dispatch_workgroups((num_bins_total + 63) / 64, 1, 1);
        }
    }

    if num_prefix_sum_steps > 1 && num_prefix_sum_steps % 2 == 0 {
        encoder.copy_buffer_to_buffer(
            &*sh.bin_offset_tmp_buf,
            0,
            &*sh.bin_offset_buf,
            0,
            ((sh.num_bins + 1) * std::mem::size_of::<u32>() as u32) as u64,
        );
    }

    // Step 4: Clear bin sizes again (for sort counting)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_clear_bin_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&*sh.sort_clear_bin_pipeline);
        pass.set_bind_group(0, &*sh.sort_particles_data_bind_groups[src_idx], &[]);
        pass.set_bind_group(1, &*sh.sort_particles_params_bind_group, &[]);
        pass.dispatch_workgroups((num_bins_total + 63) / 64, 1, 1);
    }

    // Step 5: Sort particles from src to dst buffer
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_particles_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&*sh.sort_particles_pipeline);
        pass.set_bind_group(0, &*sh.sort_particles_data_bind_groups[src_idx], &[]);
        pass.set_bind_group(1, &*sh.sort_particles_params_bind_group, &[]);
        pass.dispatch_workgroups((NUM_PARTICLES + 63) / 64, 1, 1);
    }

    // Now dst_idx buffer has sorted particles with valid bin_offset indices
    // Swap buffer index so dst becomes the new current
    model.current_buffer_idx.store(dst_idx, Ordering::Relaxed);

    // Particle simulation pass (runs on destination buffer which now has sorted particles)
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&*model.compute.particle_pipeline);
        pass.set_bind_group(0, &*model.compute.particle_bind_groups[dst_idx], &[]);
        pass.dispatch_workgroups((NUM_PARTICLES + 63) / 64, 1, 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let need_audio = model.last_buffer_level < model.request_threshold;

        // Read back audio from previous frame if there was any
        let prev_staging_idx = model.last_audio_staging_idx.load(Ordering::Relaxed);
        if prev_staging_idx >= 0 {
            let read_idx = prev_staging_idx as usize;
            let read_buf = model.compute.audio_staging_bufs[read_idx].clone();
            let read_buf_for_callback = read_buf.clone();
            let audio_producer = model.audio_producer.clone();
            let audio_buf_size = audio_buf_size as usize;

            read_buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let data = read_buf_for_callback.slice(..).get_mapped_range();
                    let floats = bytemuck::cast_slice::<u8, f32>(&data[..audio_buf_size]);
                    if let Ok(mut producer) = audio_producer.lock() {
                        producer.push_slice(floats);
                    }
                    drop(data);
                    read_buf_for_callback.unmap();
                }
            });
        }

        if need_audio {
            // Audio synthesis pass (uses destination buffer)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("audio_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&*model.compute.audio_pipeline);
                pass.set_bind_group(0, &*model.compute.audio_bind_groups[dst_idx], &[]);
                pass.dispatch_workgroups((CHUNK_SIZE + 63) / 64, 1, 1);
            }

            // Phase update pass (uses destination buffer)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("phase_update_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&*model.compute.phase_update_pipeline);
                pass.set_bind_group(0, &*model.compute.phase_update_bind_groups[dst_idx], &[]);
                pass.dispatch_workgroups((NUM_PARTICLES + 63) / 64, 1, 1);
            }

            let write_idx = if prev_staging_idx < 0 {
                0
            } else {
                (prev_staging_idx as usize + 1) % NUM_AUDIO_STAGING_BUFS
            };

            encoder.copy_buffer_to_buffer(
                &*model.compute.audio_out_buf,
                0,
                &*model.compute.audio_staging_bufs[write_idx],
                0,
                audio_buf_size,
            );

            model.last_audio_staging_idx.store(write_idx as i8, Ordering::Relaxed);
        } else {
            model.last_audio_staging_idx.store(-1, Ordering::Relaxed);
        }
    }

    // ==================== Trail-based Rendering (Ping-Pong Reprojection) ====================
    // Read from previous trail texture, reproject + decay, write to current trail texture
    let trail_prev = model.trail.ping_pong_idx.load(Ordering::Relaxed);
    let trail_curr = 1 - trail_prev;
    let is_first_frame = model.frame_count <= 1;

    // Step 1: Reproject pass - sample previous trail texture at reprojected UVs, write to current
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reproject_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &*model.trail.texture_views[trail_curr],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Skip reproject on first frame (both textures are empty)
        if !is_first_frame {
            render_pass.set_pipeline(&*model.trail.reproject_pipeline);
            render_pass.set_bind_group(0, &*model.trail.reproject_bind_groups[trail_prev], &[]);
            render_pass.draw(0..3, 0..1);
        }
    }

    // Step 2: Particle pass - render particles to current trail texture with additive blending
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particle_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &*model.trail.texture_views[trail_curr],
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,  // Preserve reprojected content
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&*model.render.pipeline);
        render_pass.set_bind_group(0, &*model.render.bind_groups[dst_idx], &[]);
        render_pass.set_vertex_buffer(0, model.render.vertex_buffer.slice(..));
        render_pass.set_index_buffer(model.render.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..NUM_PARTICLES);
    }

    // Step 3: Blit pass - copy current trail texture to screen
    {
        let mut render_pass = wgpu::RenderPassBuilder::new()
            .color_attachment(frame.resolve_target_view().unwrap(), |color| color)
            .begin(&mut encoder);

        render_pass.set_pipeline(&*model.trail.blit_pipeline);
        render_pass.set_bind_group(0, &*model.trail.blit_bind_groups[trail_curr], &[]);
        render_pass.draw(0..3, 0..1);
    }

    // Swap trail ping-pong and save viewport for next frame's reprojection
    model.trail.ping_pong_idx.store(trail_curr, Ordering::Relaxed);
    let mut prev = model.prev_viewport.lock().unwrap();
    *prev = [model.current_x0, model.current_x1, model.current_y0, model.current_y1];
}