# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Notes

The app takes at least 20 seconds to start up due to shader compilation and GPU initialization. Consider this when reading logs or testing changes.

## Build and Run Commands

```bash
# Build (release recommended for performance)
cargo build --release

# Run the application
cargo run --release

# Debug build (faster compile, slower runtime)
cargo run

# Run tests
cargo test

# Build for WASM (requires trunk and wasm-opt)
./build-wasm.sh

# Serve the WASM build locally
python3 -m http.server 8080 --directory dist
```

### WASM Notes

- Audio is disabled in the WASM build (guarded by `#[cfg(not(target_arch = "wasm32"))]`).
- Trunk's built-in wasm-opt pass is disabled (`data-wasm-opt="0"` in index.html) because it doesn't pass the required `--enable-bulk-memory` and `--enable-nontrapping-float-to-int` flags for the WASM features emitted by Rust with LTO. The `build-wasm.sh` script runs wasm-opt manually with the correct flags.
- Requires a browser with WebGPU support (Chrome/Edge).

## Project Overview

Simplelife is a GPU-accelerated particle life simulation with real-time audio synthesis. Particles interact based on species-specific rules while GPU compute shaders generate audio from particle positions. Built with Bevy/nannou (WebGPU backend) for graphics and nannou_audio for audio streaming.

## Architecture

### Core Systems

1. **Particle Simulation** - GPU compute shader (`particle.wgsl`) handles forces, collisions, and genetic copying between particles
2. **Spatial Hashing** - GPU-based spatial acceleration structure for O(1) neighbor queries, implemented across multiple shaders (`bin_fill_size.wgsl`, `bin_prefix_sum.wgsl`, `particle_sort.wgsl`)
3. **Audio Synthesis** - GPU compute shader (`audio.wgsl`) generates stereo audio from visible particles, with phase tracking (`phase_update.wgsl`)
4. **Rendering** - Instanced particle rendering with HSL coloring (`render.wgsl`)

### Data Flow

Per frame (60 FPS):
1. Clear spatial hash bins → Count particles per bin → Prefix sum → Sort particles by bin
2. Compute particle physics (forces from neighbors via spatial hash)
3. Generate audio chunk if needed (only from viewport-visible particles)
4. Update audio oscillator phases
5. Render particles

### Audio Pipeline

GPU generates 2048-sample stereo chunks → staging buffer → lock-free ring buffer → audio callback thread. PID controller adjusts generation rate based on buffer levels to prevent dropouts.

## Key Data Structures

**Particle** (96 bytes, defined in `common.wgsl`):
- Position, velocity (2D vectors)
- Phase (audio oscillator), energy
- Species values (determine interaction forces)
- Alpha/interaction parameters

**SimParams/AudioParams** in `main.rs` configure simulation constants, spatial hash grid dimensions, audio sample rate (22,050 Hz), chunk size, etc.

**Model** must derive `Clone` - this is required by the nannou render function signature.

## Shader Files

| File | Purpose |
|------|---------|
| `common.wgsl` | Shared structs, hash functions, audio helpers |
| `particle.wgsl` | Main physics simulation |
| `audio.wgsl` | Audio waveform synthesis |
| `phase_update.wgsl` | Audio phase advancement |
| `render.wgsl` | Vertex/fragment shaders for rendering |
| `bin_*.wgsl`, `particle_sort.wgsl` | Spatial hash construction |

## Controls

- **WASD** - Pan viewport
- **E/F** - Zoom in/out
- **R** - Reset viewport to full bounds

## Dependencies

- `nannou`/`nannou_audio` - From git branch `bevy-refactor` (uses bevy 0.17 internally)
- `ringbuf` - Lock-free audio buffer
- `bytemuck` - GPU memory casting
