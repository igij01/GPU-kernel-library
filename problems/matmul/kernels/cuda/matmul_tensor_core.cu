// matmul_tensor_core.cu — Matrix multiplication via WMMA tensor-core intrinsics.
//
// C = A @ B   A:(M,K) fp16, B:(K,N) fp16 → C:(M,N) fp32
//
// Design: C++ templates instead of macros.
//   - WMMA fragment dimensions are fixed: WMMA_M=16, WMMA_N=16, WMMA_K=16
//     (the only shape supported for fp16 → fp32 on all SM ≥ 70 hardware).
//   - BLOCK_TILE controls the output tile owned by each thread block:
//       block handles a (BLOCK_TILE × BLOCK_TILE) patch of C.
//     Each warp within the block is responsible for one 16×16 output patch,
//     so the number of warps per block is (BLOCK_TILE/16)².
//   - The extern "C" entry point instantiates the template with the
//     compile-time BLOCK_SIZE injected via -DBLOCK_SIZE=<value>.
//
// Thread layout: blockDim = ( (BLOCK_SIZE/16)² × 32, 1 ).
// Grid layout  : ( ceil(N/BLOCK_SIZE), ceil(M/BLOCK_SIZE) ).
//
// Preconditions (asserted at compile time):
//   - BLOCK_SIZE must be a multiple of 16.
//   - M, N, K must be multiples of 16 (no boundary padding).

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// ---------------------------------------------------------------------------
// Templated implementation (device-side)
// ---------------------------------------------------------------------------

template <int BLOCK_TILE>
__device__ void matmul_tensor_impl(
    const __half* __restrict__ A,   // (M, K) row-major
    const __half* __restrict__ B,   // (K, N) row-major
    float*        __restrict__ C,   // (M, N) row-major
    int M, int N, int K
) {
    static_assert(BLOCK_TILE % WMMA_M == 0,
                  "BLOCK_TILE must be a multiple of WMMA_M (16)");

    // How many WMMA tiles span one side of the block tile.
    constexpr int WARPS_PER_ROW = BLOCK_TILE / WMMA_N;

    // Map 1-D thread index → warp index → (warp_m, warp_n) position.
    const int warp_id = threadIdx.x / warpSize;
    const int warp_m  = warp_id / WARPS_PER_ROW;
    const int warp_n  = warp_id % WARPS_PER_ROW;

    // Top-left corner of this warp's 16×16 output tile.
    const int row_start = blockIdx.y * BLOCK_TILE + warp_m * WMMA_M;
    const int col_start = blockIdx.x * BLOCK_TILE + warp_n * WMMA_N;

    // Guard: blocks at the edge of the matrix that have no work.
    if (row_start >= M || col_start >= N) return;

    // Accumulator fragment (fp32).
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Sweep over K in steps of WMMA_K (16).
    for (int k = 0; k + WMMA_K <= K; k += WMMA_K) {
        wmma::fragment<wmma::matrix_a,
                       WMMA_M, WMMA_N, WMMA_K,
                       __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b,
                       WMMA_M, WMMA_N, WMMA_K,
                       __half, wmma::row_major> b_frag;

        // Load A[row_start : row_start+16, k : k+16].
        wmma::load_matrix_sync(a_frag, A + row_start * K + k, K);
        // Load B[k : k+16, col_start : col_start+16].
        wmma::load_matrix_sync(b_frag, B + k * N + col_start, N);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Write the 16×16 result tile to C.
    wmma::store_matrix_sync(
        C + row_start * N + col_start,
        acc_frag, N,
        wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// extern "C" entry point — instantiates the template with the compile-time
// BLOCK_SIZE value.  The framework passes BLOCK_SIZE via -DBLOCK_SIZE=<N>.
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_tensor_core(
    const __half* A,
    const __half* B,
    float*        C,
    int M, int N, int K
) {
    matmul_tensor_impl<BLOCK_SIZE>(A, B, C, M, N, K);
}
