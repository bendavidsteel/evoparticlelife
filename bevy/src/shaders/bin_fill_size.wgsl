@group(0) @binding(0) var<storage, read> particles : array<Particle>;

@group(1) @binding(0) var<uniform> params : SimParams;

@group(2) @binding(0) var<storage, read_write> binSize : array<atomic<u32>>;

@compute @workgroup_size(64)
fn clearBinSize(@builtin(global_invocation_id) id : vec3u)
{
    if (id.x >= arrayLength(&binSize)) {
        return;
    }

    atomicStore(&binSize[id.x], 0u);
}

@compute @workgroup_size(64)
fn fillBinSize(@builtin(global_invocation_id) id : vec3u)
{
    if (id.x >= arrayLength(&particles)) {
        return;
    }

    let particle = particles[id.x];

    let binIndex = getBinInfo(particle.pos, params.map_x0, params.map_y0, params.bin_size, params.num_bins_x, params.num_bins_y).binIndex;

    atomicAdd(&binSize[binIndex + 1], 1u);
}
