// Vector addition: C = A + B
// BLOCK_SIZE is injected as a -D preprocessor define by the CUDA compiler.

extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
