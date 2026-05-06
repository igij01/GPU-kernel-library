// matmul_core.cu — Tiled matrix multiplication using shared memory.
//
// C = A @ B   A:(M,K) InputT, B:(K,N) InputT → C:(M,N) AccT
//
// Design: C++ templates instead of macros.
//   - matmul_core_impl<TILE, InputT, AccT> is a __device__ function templated
//     on the tile size and on the input / accumulator element types.
//   - Inputs are loaded and cast to AccT before entering shared memory, so
//     the accumulation loop always runs in AccT precision.
//   - The extern "C" __global__ entry point matmul_core() instantiates the
//     template with BLOCK_SIZE (injected via -DBLOCK_SIZE=<value>) and the
//     concrete types __half / float.  No preprocessor macros are used for
//     the tile size or the types.
//
// Thread layout: blockDim = (BLOCK_SIZE, BLOCK_SIZE).
// Grid layout  : (ceil(N/BLOCK_SIZE), ceil(M/BLOCK_SIZE)).

#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Templated implementation (device-side, not callable as a kernel)
// ---------------------------------------------------------------------------

template <int TILE, typename InputT, typename AccT>
__device__ void matmul_core_impl(
    const InputT* __restrict__ A,   // (M, K) row-major
    const InputT* __restrict__ B,   // (K, N) row-major
    AccT*         __restrict__ C,   // (M, N) row-major
    int M, int N, int K
) {
    // Shared-memory tiles hold values already promoted to AccT, so the
    // inner dot-product accumulates entirely in AccT precision.
    __shared__ AccT sA[TILE][TILE];
    __shared__ AccT sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    AccT acc = AccT(0);

    // Sweep over K-tiles.  Each iteration loads one (TILE×TILE) strip of A
    // and one (TILE×TILE) strip of B into shared memory.
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        // static_cast handles InputT → AccT (e.g. __half → float).
        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? static_cast<AccT>(A[row * K + a_col]) : AccT(0);
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? static_cast<AccT>(B[b_row * N + col]) : AccT(0);
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
// Template entry point — instantiated by CuPy's name-expression mechanism.
//
// Template parameters:
//   TILE    — tile/block size (integer, from config_space)
//   InputT  — input element type (mapped from problem dtype via type_args)
//
// The accumulator type is always float (fp32).  Mixed input/output type
// support (e.g. fp16 input → fp32 accumulation) is the common case for
// matmul; AccT is not exposed as a template parameter because the framework's
// type_args binds all listed params to a single dtype.
// ---------------------------------------------------------------------------

template <int TILE, typename InputT>
__global__ void matmul_core(
    const InputT* __restrict__ A,
    const InputT* __restrict__ B,
    float*        __restrict__ C,
    int M, int N, int K
) {
    matmul_core_impl<TILE, InputT, float>(A, B, C, M, N, K);
}
