//  sme_kernels.h
//  Hand-written Arm SME2 matrix kernels (plus NEON baselines) for Apple A19 / M4-class CPUs.
//
//  Layout conventions (all row-major):
//    C[M x N] = A[M x K] * B[K x N] (+ bias[N]) (optionally ReLU)
//
//  Both operands are "packed" into panels before the SME kernel runs:
//    - Weights (B) are packed once, at model load.
//    - Activations (A) are packed per call (cheap, O(M*K)).
//  Packed buffers are zero padded, so the inner loop never needs edge predicates.

#ifndef SME_KERNELS_H
#define SME_KERNELS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------- Capability queries (safe to call on any device) ----------

/// True if the CPU reports FEAT_SME.
bool sme_cpu_has_sme(void);
/// True if the CPU reports FEAT_SME2. All SME kernels below require this.
bool sme_cpu_has_sme2(void);
/// Streaming vector length in bytes (64 on A19 / M4 = 512-bit). 0 if no SME.
int sme_streaming_vector_bytes(void);
/// Model identifier, e.g. "iPhone18,1". Writes into buf.
void sme_device_model(char *buf, size_t len);

// ---------- FP32 GEMM ----------

/// Number of floats needed for a packed A (activations) buffer.
size_t sme_f32_packed_a_count(int M, int K);
/// Number of floats needed for a packed B (weights) buffer.
size_t sme_f32_packed_b_count(int K, int N);
void sme_f32_pack_a(const float *A, int M, int K, float *Ap);
void sme_f32_pack_b(const float *B, int K, int N, float *Bp);

/// C = Ap * Bp (+bias) (ReLU if relu != 0). bias may be NULL. ldc = row stride of C.
/// Runs in SME streaming mode using the ZA tile array (FMOPA outer products).
void sme_f32_gemm_packed(int M, int N, int K,
                         const float *Ap, const float *Bp,
                         const float *bias, int relu,
                         float *C, int ldc);

/// NEON (non-streaming) baseline that reads the same packed buffers.
void neon_f32_gemm_packed(int M, int N, int K,
                          const float *Ap, const float *Bp,
                          const float *bias, int relu,
                          float *C, int ldc);

// ---------- INT8 GEMM (int32 accumulation) ----------

size_t sme_s8_packed_a_bytes(int M, int K);
size_t sme_s8_packed_b_bytes(int K, int N);
void sme_s8_pack_a(const int8_t *A, int M, int K, int8_t *Ap);
void sme_s8_pack_b(const int8_t *B, int K, int N, int8_t *Bp);

/// C(int32) = Ap(int8) * Bp(int8). Uses the 4-way SMOPA widening outer product.
void sme_s8_gemm_packed(int M, int N, int K,
                        const int8_t *Ap, const int8_t *Bp,
                        int32_t *C, int ldc);

/// NEON SDOT baseline on the same packed layout.
void neon_s8_gemm_packed(int M, int N, int K,
                         const int8_t *Ap, const int8_t *Bp,
                         int32_t *C, int ldc);

// ---------- Reference (scalar, for correctness checks) ----------

void ref_f32_gemm(int M, int N, int K, const float *A, const float *B,
                  const float *bias, int relu, float *C);
void ref_s8_gemm(int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C);

// ---------- Timing ----------

/// Monotonic time in nanoseconds.
uint64_t sme_now_ns(void);

#ifdef __cplusplus
}
#endif

#endif /* SME_KERNELS_H */
