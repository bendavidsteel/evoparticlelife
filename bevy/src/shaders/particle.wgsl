@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read> bin_offset: array<u32>;

fn hash( x: u32 ) -> u32 {
    var u = x;
    u += ( u << 10u );
    u ^= ( u >>  6u );
    u += ( u <<  3u );
    u ^= ( u >> 11u );
    u += ( u << 15u );
    return u;
}

fn random( f: f32 ) -> f32 {
    let mantissaMask = 0x007FFFFFu;
    let one          = 0x3F800000u;

    var h = hash( bitcast<u32>(f) );
    h &= mantissaMask;
    h |= one;
    
    let r2 = bitcast<f32>( h );
    return r2 - 1.0;
}

fn random2( v: vec2f ) -> f32 { return random( v.x + v.y * 57.0 ); }
fn random3( v: vec3f ) -> f32 { return random( v.x + v.y * 57.0 + v.z * 113.0 ); }
fn random4( v: vec4f ) -> f32 { return random( v.x + v.y * 57.0 + v.z * 113.0 + v.w * 197.0 ); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.num_particles {
        return;
    }

    var particle = particles[id.x];

    let binInfo = getBinInfo(particle.pos, params.map_x0, params.map_y0, params.bin_size, params.num_bins_x, params.num_bins_y);

    var binXMin = i32(binInfo.binX) - 1;
    var binYMin = i32(binInfo.binY) - 1;

    var binXMax = i32(binInfo.binX) + 1;
    var binYMax = i32(binInfo.binY) + 1;

    let width = params.map_x1 - params.map_x0;
    let height = params.map_y1 - params.map_y0;

    var totalForce = vec2f(0.0, 0.0);
    var totalEnergy = 0.0;
    var totalInteraction = vec2f(0.0, 0.0);
    var numInteractions = 0u;

    let copyBinXRand = random3(vec3f(params.time, particle.pos.x, particle.pos.y));
    let copyBinYRand = random4(vec4f(params.time, particle.pos.y, particle.pos.x, particle.vel.x));
    let copyBinXOffset = i32(floor(copyBinXRand * 3.0)) - 1;
    let copyBinYOffset = i32(floor(copyBinYRand * 3.0)) - 1;

    for (var binX = binXMin; binX <= binXMax; binX += 1) {
        for (var binY = binYMin; binY <= binYMax; binY += 1) {
            var realBinX = u32((binX + i32(params.num_bins_x)) % i32(params.num_bins_x));
            var realBinY = u32((binY + i32(params.num_bins_y)) % i32(params.num_bins_y));

            let binIndex = realBinY * params.num_bins_x + realBinX;
            let binStart = bin_offset[binIndex];
            let binEnd = bin_offset[binIndex + 1];

            for (var j = binStart; j < binEnd; j += 1) {
                let other = particles[j];

                let alpha = dot(particle.alpha, other.species) * params.max_force_strength;

                var r = other.pos - particle.pos;

                if (abs(r.x) >= width * 0.5) {
                    r.x -= sign(r.x) * width;
                }

                if (abs(r.y) >= height * 0.5) {
                    r.y -= sign(r.y) * height;
                }

                let d = length(r);
                if (d > 0.0 && d < params.radius) {
                    let n = r / d;

                    var f = 0.0;
                    var e = 0.0;
                    let forceRegion = params.radius - params.collision_radius;
                    let r_mid = params.collision_radius + forceRegion * 0.5;
                    if (d < params.collision_radius) {
                        let dd = params.collision_radius - d;
                        f = dd * -params.collision_strength;
                        e = 0.5 * (f * dd - alpha * forceRegion);
                    } else if (d >= params.collision_radius && d < r_mid) {
                        let dd = d - params.collision_radius;
                        f = 2 * alpha * dd / forceRegion;
                        e = 0.5 * (f * dd - alpha * forceRegion);
                    } else {
                        let dm = d - params.radius;
                        f = -2 * alpha * dm / forceRegion;
                        e = 0.5 * f * dm;
                    }

                    totalForce += f * n;
                    totalEnergy += e;

                    totalInteraction += 0.01 * (other.species - particle.species);
                }

                if (binX == i32(binInfo.binX) + copyBinXOffset && binY == i32(binInfo.binY) + copyBinYOffset) {
                    // check if we should copy this other particle's genes
                    if (d < params.copy_radius) {
                        let speciesCosSim = dot(particle.species, other.species) / (length(particle.species) * length(other.species) + 1e-6);

                        if (speciesCosSim > params.copy_cos_sim_threshold) {
                            let p = random3(vec3f(params.time, particle.pos.x, other.pos.x));
                            if (p < params.copy_probability) {
                                particle.species = other.species;
                                particle.alpha = other.alpha;
                            }
                        }
                    }
                }
            }
        }
    }

    let mu = pow(0.5, params.dt / params.friction);
    particle.vel *= mu;

    particle.vel += totalForce * params.dt / params.mass;

    // Update position based on velocity
    particle.pos += particle.vel * params.dt;

    // Wrap positions to stay in [-1, 1] range
    let map_min = vec2<f32>(params.map_x0, params.map_y0);
    let map_max = vec2<f32>(params.map_x1, params.map_y1);
    let map_size = map_max - map_min;
    particle.pos -= floor((particle.pos - map_min) / map_size) * map_size;

    // Update energy based on energy func
    particle.energy = totalEnergy;

    // Update interaction
    particle.interaction = totalInteraction;

    particles[id.x] = particle;
}