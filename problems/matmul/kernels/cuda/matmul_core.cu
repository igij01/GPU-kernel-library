// matmul_core.cu — Tiled matrix multiplication using shared memory.
//
// C = A @ B   A:(M,K) fp16, B:(K,N) fp16 → C:(M,N) fp32
//
// Design: C++ templates instead of macros.
//   - matmul_core_impl<TILE> is a __device__ function templated on the
//     shared-memory tile size.
//   - The extern "C" __global__ entry point matmul_core() instantiates the
//     template using BLOCK_SIZE, which is injected at compile time via
//     -DBLOCK_SIZE=<value>.  No preprocessor macros are used for the tile.
//
// Thread layout: blockDim = (BLOCK_SIZE, BLOCK_SIZE).
// Grid layout  : (ceil(N/BLOCK_SIZE), ceil(M/BLOCK_SIZE)).

#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Templated implementation (device-side, not callable as a kernel)
// ---------------------------------------------------------------------------

template <int TILE>
__device__ void matmul_core_impl(
    const __half* __restrict__ A,   // (M, K) row-major
    const __half* __restrict__ B,   // (K, N) row-major
    float*        __restrict__ C,   // (M, N) row-major
    int M, int N, int K
) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // Sweep over K-tiles.  Each iteration loads one (TILE×TILE) strip of A
    // and one (TILE×TILE) strip of B into shared memory.
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? __half2float(A[row * K + a_col]) : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? __half2float(B[b_row * N + col]) : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// extern "C" entry point — instantiates the template with the compile-time
// BLOCK_SIZE value.  The framework passes BLOCK_SIZE via -DBLOCK_SIZE=<N>.
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_core(
    const __half* A,
    const __half* B,
    float*        C,
    int M, int N, int K
) {
    matmul_core_impl<BLOCK_SIZE>(A, B, C, M, N, K);
}
