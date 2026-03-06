@group(0) @binding(0) var<storage, read> source : array<Particle>;
@group(0) @binding(1) var<storage, read_write> destination : array<Particle>;
@group(0) @binding(2) var<storage, read> binOffset : array<u32>;
@group(0) @binding(3) var<storage, read_write> binSize : array<atomic<u32>>;

@group(1) @binding(0) var<uniform> params : SimParams;

@compute @workgroup_size(64)
fn clearBinSize(@builtin(global_invocation_id) id : vec3u)
{
    if (id.x >= arrayLength(&binSize)) {
        return;
    }

    atomicStore(&binSize[id.x], 0u);
}

@compute @workgroup_size(64)
fn sortParticles(@builtin(global_invocation_id) id : vec3u)
{
    if (id.x >= arrayLength(&source)) {
        return;
    }

    let particle = source[id.x];

    let binIndex = getBinInfo(particle.pos, params.map_x0, params.map_y0, params.bin_size, params.num_bins_x, params.num_bins_y).binIndex;

    let newParticleIndex = binOffset[binIndex] + atomicAdd(&binSize[binIndex], 1);
    destination[newParticleIndex] = particle;
}
