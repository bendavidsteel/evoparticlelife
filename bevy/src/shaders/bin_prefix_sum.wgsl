@group(0) @binding(0) var<storage, read> source : array<u32>;
@group(0) @binding(1) var<storage, read_write> destination : array<u32>;

const WORKGROUP_SIZE = 256u;
const BLOCK_SIZE = 4u;  // elements per thread

var<workgroup> partial_sums : array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(256)
fn prefixSum(@builtin(local_invocation_id) lid : vec3u)
{
    let n = arrayLength(&source);
    let tid = lid.x;
    let base = tid * BLOCK_SIZE;

    // Phase 1: Each thread loads BLOCK_SIZE elements and computes a local inclusive prefix sum.
    // Store the per-element sums in registers (we'll write them out in phase 3).
    var local_vals = array<u32, 4>();
    var running_sum = 0u;

    for (var i = 0u; i < BLOCK_SIZE; i++) {
        let idx = base + i;
        if (idx < n) {
            running_sum += source[idx];
        }
        local_vals[i] = running_sum;
    }

    // The partial sum for this thread's block is the total of its elements.
    partial_sums[tid] = running_sum;
    workgroupBarrier();

    // Phase 2: Hillis-Steele inclusive scan of the 256 partial sums.
    for (var step = 1u; step < WORKGROUP_SIZE; step <<= 1u) {
        var val = partial_sums[tid];
        if (tid >= step) {
            val += partial_sums[tid - step];
        }
        workgroupBarrier();
        partial_sums[tid] = val;
        workgroupBarrier();
    }

    // Phase 3: Each thread adds the prefix from the previous block to its local values
    // and writes to destination.
    let prefix = select(0u, partial_sums[tid - 1u], tid > 0u);

    for (var i = 0u; i < BLOCK_SIZE; i++) {
        let idx = base + i;
        if (idx < n) {
            destination[idx] = local_vals[i] + prefix;
        }
    }
}
