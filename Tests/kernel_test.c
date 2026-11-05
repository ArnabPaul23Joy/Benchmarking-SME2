// Host-side test for the kernels (runs on any SME2 Mac, e.g. M4):
//   xcrun clang -O3 -framework Accelerate -I SME2Inference/Kernels \
//       SME2Inference/Kernels/sme_kernels.c Tests/kernel_test.c -o /tmp/kt && /tmp/kt
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sme_kernels.h"

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

static int check_f32(int M, int N, int K) {
    float *A = malloc(sizeof(float)*M*K), *B = malloc(sizeof(float)*K*N), *bias = malloc(sizeof(float)*N);
    float *C1 = malloc(sizeof(float)*M*N), *C2 = malloc(sizeof(float)*M*N), *C3 = malloc(sizeof(float)*M*N);
    for (int i = 0; i < M*K; i++) A[i] = frand();
    for (int i = 0; i < K*N; i++) B[i] = frand();
    for (int i = 0; i < N; i++) bias[i] = frand();
    float *Ap = malloc(sizeof(float)*sme_f32_packed_a_count(M,K)), *Bp = malloc(sizeof(float)*sme_f32_packed_b_count(K,N));
    sme_f32_pack_a(A, M, K, Ap); sme_f32_pack_b(B, K, N, Bp);
    ref_f32_gemm(M, N, K, A, B, bias, 1, C1);
    sme_f32_gemm_packed(M, N, K, Ap, Bp, bias, 1, C2, N);
    neon_f32_gemm_packed(M, N, K, Ap, Bp, bias, 1, C3, N);
    double e2 = 0, e3 = 0;
    for (int i = 0; i < M*N; i++) { e2 = fmax(e2, fabs(C1[i]-C2[i])); e3 = fmax(e3, fabs(C1[i]-C3[i])); }
    int ok = e2 < 1e-3 * K && e3 < 1e-3 * K;
    printf("f32 %4dx%4dx%4d  sme err %.2e  neon err %.2e  %s\n", M, N, K, e2, e3, ok ? "OK" : "FAIL");
    free(A); free(B); free(bias); free(C1); free(C2); free(C3); free(Ap); free(Bp);
    return ok;
}

static int check_s8(int M, int N, int K) {
    int8_t *A = malloc(M*K), *B = malloc(K*N);
    int32_t *C1 = malloc(4*M*N), *C2 = malloc(4*M*N), *C3 = malloc(4*M*N);
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)(rand() % 255 - 127);
    int8_t *Ap = malloc(sme_s8_packed_a_bytes(M,K)), *Bp = malloc(sme_s8_packed_b_bytes(K,N));
    sme_s8_pack_a(A, M, K, Ap); sme_s8_pack_b(B, K, N, Bp);
    ref_s8_gemm(M, N, K, A, B, C1);
    sme_s8_gemm_packed(M, N, K, Ap, Bp, C2, N);
    neon_s8_gemm_packed(M, N, K, Ap, Bp, C3, N);
    int bad2 = 0, bad3 = 0;
    for (int i = 0; i < M*N; i++) { bad2 += C1[i] != C2[i]; bad3 += C1[i] != C3[i]; }
    printf("s8  %4dx%4dx%4d  sme mismatches %d  neon mismatches %d  %s\n", M, N, K, bad2, bad3, (!bad2 && !bad3) ? "OK" : "FAIL");
    free(A); free(B); free(C1); free(C2); free(C3); free(Ap); free(Bp);
    return !bad2 && !bad3;
}

static void bench(int n) {
    const int M = n, N = n, K = n;
    float *A = malloc(sizeof(float)*M*K), *B = malloc(sizeof(float)*K*N), *C = malloc(sizeof(float)*M*N);
    for (int i = 0; i < M*K; i++) A[i] = frand();
    for (int i = 0; i < K*N; i++) B[i] = frand();
    float *Ap = malloc(sizeof(float)*sme_f32_packed_a_count(M,K)), *Bp = malloc(sizeof(float)*sme_f32_packed_b_count(K,N));
    sme_f32_pack_a(A, M, K, Ap); sme_f32_pack_b(B, K, N, Bp);
    int8_t *A8 = malloc(M*K), *B8 = malloc(K*N); int32_t *C32 = malloc(4*M*N);
    for (int i = 0; i < M*K; i++) A8[i] = (int8_t)(rand()%255-127);
    for (int i = 0; i < K*N; i++) B8[i] = (int8_t)(rand()%255-127);
    int8_t *Ap8 = malloc(sme_s8_packed_a_bytes(M,K)), *Bp8 = malloc(sme_s8_packed_b_bytes(K,N));
    sme_s8_pack_a(A8, M, K, Ap8); sme_s8_pack_b(B8, K, N, Bp8);
    double flops = 2.0 * M * N * K;
    #define TIME(label, stmt) { int it = 0; uint64_t t0 = sme_now_ns(), t; do { stmt; it++; t = sme_now_ns(); } while (t - t0 < 300000000ull); \
        double s = (t - t0) / 1e9 / it; printf("  n=%4d %-22s %8.3f ms  %8.1f G(FL)OPS\n", n, label, s*1e3, flops/s/1e9); }
    TIME("SME2 f32 FMOPA", sme_f32_gemm_packed(M,N,K,Ap,Bp,NULL,0,C,N));
    TIME("NEON f32", neon_f32_gemm_packed(M,N,K,Ap,Bp,NULL,0,C,N));
    TIME("Accelerate sgemm", cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,1,A,K,B,N,0,C,N));
    TIME("SME2 s8 SMOPA", sme_s8_gemm_packed(M,N,K,Ap8,Bp8,C32,N));
    TIME("NEON s8 SDOT", neon_s8_gemm_packed(M,N,K,Ap8,Bp8,C32,N));
}

int main(void) {
    char model[64]; sme_device_model(model, sizeof model);
    printf("device %s  SME %d  SME2 %d  SVL %d bytes\n", model, sme_cpu_has_sme(), sme_cpu_has_sme2(), sme_streaming_vector_bytes());
    int ok = 1;
    int sizes[][3] = {{1,1,1},{1,10,7},{5,17,3},{31,33,65},{32,32,32},{64,96,128},{100,37,250},{129,257,513}};
    for (unsigned i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        ok &= check_f32(sizes[i][0], sizes[i][1], sizes[i][2]);
        ok &= check_s8(sizes[i][0], sizes[i][1], sizes[i][2]);
    }
    printf(ok ? "ALL CORRECT\n" : "FAILURES\n");
    bench(256); bench(512); bench(1024);
    return !ok;
}
