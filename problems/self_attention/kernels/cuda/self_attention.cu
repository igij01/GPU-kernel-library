// Single-head self-attention: O = softmax(Q K^T / sqrt(D)) V
//
// Grid : (S,)          — one block per query row
// Block: (BLOCK_SIZE,) — threads collaborate on a single output row
//
// Shared memory: S floats for the attention score vector.
// Restriction  : S <= 1024 (hard-coded shared-mem buffer size).
//
// BLOCK_SIZE is injected as a -D preprocessor define.

// rsqrtf is a CUDA built-in — no include needed.

extern "C"
__global__ void self_attention(
    const float* __restrict__ Q,   // (S, D)
    const float* __restrict__ K,   // (S, D)
    const float* __restrict__ V,   // (S, D)
          float* __restrict__ O,   // (S, D)
    int S,
    int D
) {
    int i = blockIdx.x;            // query row this block handles

    // Shared storage for the S attention scores of query row i.
    __shared__ float scores[1024]; // supports S up to 1024

    float inv_sqrt_d = rsqrtf((float)D);

    // ---------------------------------------------------------------
    // Step 1 : compute raw scores[j] = dot(Q[i], K[j]) / sqrt(D)
    //          Threads split the j dimension.
    // ---------------------------------------------------------------
    for (int j = threadIdx.x; j < S; j += BLOCK_SIZE) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[i * D + d] * K[j * D + d];
        }
        scores[j] = dot * inv_sqrt_d;
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // Step 2 : softmax (thread 0 does the serial reduction; naive but
    //          correct — the focus here is kernel structure, not speed)
    // ---------------------------------------------------------------
    if (threadIdx.x == 0) {
        float max_val = scores[0];
        for (int j = 1; j < S; ++j) {
            if (scores[j] > max_val) max_val = scores[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < S; ++j) {
            scores[j] = expf(scores[j] - max_val);
            sum += scores[j];
        }
        for (int j = 0; j < S; ++j) {
            scores[j] /= sum;
        }
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // Step 3 : O[i, d] = sum_j( scores[j] * V[j, d] )
    //          Threads split the d (output) dimension.
    // ---------------------------------------------------------------
    for (int d = threadIdx.x; d < D; d += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int j = 0; j < S; ++j) {
            acc += scores[j] * V[j * D + d];
        }
        O[i * D + d] = acc;
    }
}
