//  sme_kernels.c
//  Arm SME2 GEMM kernels for Apple silicon (A19 Pro in iPhone 17 Pro, M4, ...).
//
//  How the SME kernels work
//  ------------------------
//  SME adds a 2-D register array called ZA. With a 512-bit streaming vector length (SVL)
//  ZA holds four 16x16 fp32 tiles (ZA0..ZA3). One FMOPA instruction does a 16x16 outer
//  product (256 FMAs) and accumulates it into a tile.
//
//  For each 32x32 block of C we keep all four tiles busy:
//
//              B cols 0-15   B cols 16-31
//    A rows 0-15   [ ZA0 ]      [ ZA1 ]
//    A rows 16-31  [ ZA2 ]      [ ZA3 ]
//
//  and for every k: load 2 vectors of A (a column slice), 2 vectors of B (a row slice)
//  with one SME2 multi-vector load each, then issue 4 FMOPAs. When K is exhausted the
//  tiles are read back row by row, bias/ReLU applied, and written to C.
//
//  Everything is vector-length agnostic: on hardware with a different SVL the panel
//  size changes automatically.
//
//  Streaming mode rules that matter here:
//    * __arm_locally_streaming: the compiler emits SMSTART/SMSTOP around the function.
//    * __arm_new("za"): the function owns fresh ZA state.
//    * Inside streaming code, avoid calling normal functions (memcpy, printf, ...) and
//      avoid NEON: Apple's cores do not implement FEAT_SME_FA64, so most Advanced SIMD
//      instructions are illegal in streaming mode. Packing is therefore done outside.

#include "sme_kernels.h"

#include <string.h>
#include <sys/sysctl.h>
#include <mach/mach_time.h>
#include <arm_neon.h>

#include <arm_sme.h>

// Enable SME2 code generation per function, so the rest of the app is compiled normally.
#define SME_TARGET __attribute__((target("sme2")))

// ---------------------------------------------------------------------------------------
// Capability detection
// ---------------------------------------------------------------------------------------

static int sysctl_int(const char *name) {
    int value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, NULL, 0) != 0) return 0;
    return value;
}

bool sme_cpu_has_sme(void)  { return sysctl_int("hw.optional.arm.FEAT_SME") != 0; }
bool sme_cpu_has_sme2(void) { return sysctl_int("hw.optional.arm.FEAT_SME2") != 0; }

__attribute__((target("sme")))
static uint64_t read_streaming_words(void) {
    // RDSVL works outside streaming mode; it SIGILLs on CPUs without SME, so only call
    // this after checking sme_cpu_has_sme().
    return svcntsw();
}

static uint64_t g_svl_words = 0;  // cached streaming vector length, in 32-bit words

static uint64_t svl_words(void) {
    if (g_svl_words == 0) {
        g_svl_words = sme_cpu_has_sme() ? read_streaming_words() : 16;  // 16 = fallback for NEON-only devices
    }
    return g_svl_words;
}

int sme_streaming_vector_bytes(void) {
    return sme_cpu_has_sme() ? (int)(svl_words() * 4) : 0;
}

void sme_device_model(char *buf, size_t len) {
    if (len == 0) return;
    buf[0] = 0;
    size_t size = len;
    if (sysctlbyname("hw.machine", buf, &size, NULL, 0) != 0) buf[0] = 0;
    buf[len - 1] = 0;
}

uint64_t sme_now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

// Panel width = 2 vectors (two ZA tiles side by side). 32 on a 512-bit machine.
static inline int f32_panel(void) { return (int)(2 * svl_words()); }
static inline int s8_panel(void)  { return (int)(2 * svl_words()); }  // rows per panel (each row = 4 int8 k-values)

static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

// ---------------------------------------------------------------------------------------
// FP32 packing
// ---------------------------------------------------------------------------------------
//  Ap: for each panel of P rows: K x P floats, Ap[k*P + i] = A[(m0+i)*K + k]   (A transposed)
//  Bp: for each panel of P cols: K x P floats, Bp[k*P + j] = B[k*N + n0 + j]

size_t sme_f32_packed_a_count(int M, int K) { int P = f32_panel(); return (size_t)round_up(M, P) * (size_t)K; }
size_t sme_f32_packed_b_count(int K, int N) { int P = f32_panel(); return (size_t)round_up(N, P) * (size_t)K; }

void sme_f32_pack_a(const float *A, int M, int K, float *Ap) {
    const int P = f32_panel();
    for (int m0 = 0; m0 < M; m0 += P) {
        float *dst = Ap + (size_t)(m0 / P) * K * P;
        const int rows = (M - m0) < P ? (M - m0) : P;
        if (rows < P) memset(dst, 0, sizeof(float) * (size_t)K * P);
        for (int i = 0; i < rows; i++) {
            const float *src = A + (size_t)(m0 + i) * K;
            for (int k = 0; k < K; k++) dst[(size_t)k * P + i] = src[k];
        }
    }
}

void sme_f32_pack_b(const float *B, int K, int N, float *Bp) {
    const int P = f32_panel();
    for (int n0 = 0; n0 < N; n0 += P) {
        float *dst = Bp + (size_t)(n0 / P) * K * P;
        const int cols = (N - n0) < P ? (N - n0) : P;
        for (int k = 0; k < K; k++) {
            memcpy(dst + (size_t)k * P, B + (size_t)k * N + n0, sizeof(float) * cols);
            if (cols < P) memset(dst + (size_t)k * P + cols, 0, sizeof(float) * (P - cols));
        }
    }
}

// ---------------------------------------------------------------------------------------
// FP32 SME2 kernel
// ---------------------------------------------------------------------------------------

// Read one horizontal slice of a ZA tile, add bias, optional ReLU, store with a column mask.
#define STORE_F32_SLICE(TILE, SLICE, BIASV, PRED, DST)                               \
    do {                                                                            \
        svfloat32_t v_ = svread_hor_za32_f32_m(svdup_f32(0.0f), pt, TILE, SLICE);   \
        v_ = svadd_f32_x(pt, v_, BIASV);                                            \
        if (relu) v_ = svmax_f32_x(pt, v_, svdup_f32(0.0f));                        \
        svst1_f32(PRED, DST, v_);                                                   \
    } while (0)

SME_TARGET __arm_new("za") __arm_locally_streaming
static void sme_f32_kernel(int M, int N, int K,
                           const float *Ap, const float *Bp,
                           const float *bias, int relu,
                           float *C, int ldc) {
    const int64_t vl = (int64_t)svcntw();   // fp32 lanes per vector (16 at 512-bit)
    const int64_t P  = 2 * vl;
    const svbool_t  pt = svptrue_b32();
    const svcount_t pc = svptrue_c32();

    for (int64_t m0 = 0; m0 < M; m0 += P) {
        const float *a_panel = Ap + (size_t)(m0 / P) * (size_t)K * P;

        for (int64_t n0 = 0; n0 < N; n0 += P) {
            const float *b_panel = Bp + (size_t)(n0 / P) * (size_t)K * P;
            const float *a = a_panel;
            const float *b = b_panel;

            svzero_za();

            for (int k = 0; k < K; k++) {
                // SME2 multi-vector loads: two consecutive vectors in one instruction.
                svfloat32x2_t av = svld1_f32_x2(pc, a);
                svfloat32x2_t bv = svld1_f32_x2(pc, b);
                svfloat32_t a0 = svget2(av, 0), a1 = svget2(av, 1);
                svfloat32_t b0 = svget2(bv, 0), b1 = svget2(bv, 1);
                svmopa_za32_f32_m(0, pt, pt, a0, b0);
                svmopa_za32_f32_m(1, pt, pt, a0, b1);
                svmopa_za32_f32_m(2, pt, pt, a1, b0);
                svmopa_za32_f32_m(3, pt, pt, a1, b1);
                a += P;
                b += P;
            }

            // Epilogue: ZA -> C with bias + ReLU, masking columns past N and rows past M.
            const svbool_t pn0 = svwhilelt_b32_s64(n0, (int64_t)N);
            const svbool_t pn1 = svwhilelt_b32_s64(n0 + vl, (int64_t)N);
            const svfloat32_t bias0 = bias ? svld1_f32(pn0, bias + n0)      : svdup_f32(0.0f);
            const svfloat32_t bias1 = bias ? svld1_f32(pn1, bias + n0 + vl) : svdup_f32(0.0f);

            for (int64_t i = 0; i < vl; i++) {
                const int64_t r0 = m0 + i;
                const int64_t r1 = m0 + vl + i;
                if (r0 < M) {
                    float *dst = C + (size_t)r0 * ldc + n0;
                    STORE_F32_SLICE(0, (uint32_t)i, bias0, pn0, dst);
                    STORE_F32_SLICE(1, (uint32_t)i, bias1, pn1, dst + vl);
                }
                if (r1 < M) {
                    float *dst = C + (size_t)r1 * ldc + n0;
                    STORE_F32_SLICE(2, (uint32_t)i, bias0, pn0, dst);
                    STORE_F32_SLICE(3, (uint32_t)i, bias1, pn1, dst + vl);
                }
            }
        }
    }
}

void sme_f32_gemm_packed(int M, int N, int K,
                         const float *Ap, const float *Bp,
                         const float *bias, int relu,
                         float *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0 || !sme_cpu_has_sme2()) return;
    sme_f32_kernel(M, N, K, Ap, Bp, bias, relu, C, ldc);
}

// ---------------------------------------------------------------------------------------
// FP32 NEON baseline: 8x8 register-blocked micro-kernel on the same packed panels
// ---------------------------------------------------------------------------------------

#define FMA_ROW(R, AV, LANE)                                     \
    acc[R][0] = vfmaq_laneq_f32(acc[R][0], b0, AV, LANE);        \
    acc[R][1] = vfmaq_laneq_f32(acc[R][1], b1, AV, LANE);

void neon_f32_gemm_packed(int M, int N, int K,
                          const float *Ap, const float *Bp,
                          const float *bias, int relu,
                          float *C, int ldc) {
    const int P = f32_panel();
    for (int m0 = 0; m0 < M; m0 += P) {
        const float *a_panel = Ap + (size_t)(m0 / P) * K * P;
        for (int n0 = 0; n0 < N; n0 += P) {
            const float *b_panel = Bp + (size_t)(n0 / P) * K * P;
            for (int r0 = 0; r0 < P && m0 + r0 < M; r0 += 8) {
                for (int c0 = 0; c0 < P && n0 + c0 < N; c0 += 8) {
                    float32x4_t acc[8][2];
                    for (int r = 0; r < 8; r++) { acc[r][0] = vdupq_n_f32(0); acc[r][1] = vdupq_n_f32(0); }
                    const float *a = a_panel + r0;
                    const float *b = b_panel + c0;
                    for (int k = 0; k < K; k++) {
                        const float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
                        const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4);
                        FMA_ROW(0, a0, 0) FMA_ROW(1, a0, 1) FMA_ROW(2, a0, 2) FMA_ROW(3, a0, 3)
                        FMA_ROW(4, a1, 0) FMA_ROW(5, a1, 1) FMA_ROW(6, a1, 2) FMA_ROW(7, a1, 3)
                        a += P;
                        b += P;
                    }
                    float tmp[8][8];
                    for (int r = 0; r < 8; r++) { vst1q_f32(tmp[r], acc[r][0]); vst1q_f32(tmp[r] + 4, acc[r][1]); }
                    const int rows = (M - m0 - r0) < 8 ? (M - m0 - r0) : 8;
                    const int cols = (N - n0 - c0) < 8 ? (N - n0 - c0) : 8;
                    for (int r = 0; r < rows; r++) {
                        float *dst = C + (size_t)(m0 + r0 + r) * ldc + n0 + c0;
                        for (int c = 0; c < cols; c++) {
                            float v = tmp[r][c] + (bias ? bias[n0 + c0 + c] : 0.0f);
                            dst[c] = (relu && v < 0.0f) ? 0.0f : v;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// INT8 packing
// ---------------------------------------------------------------------------------------
//  SMOPA (int8 -> int32, 4-way) computes, per tile element (i, j):
//      ZA[i][j] += sum_{t=0..3} a[4i + t] * b[4j + t]
//  so each vector holds 16 rows (or cols) x 4 consecutive k values.
//  Packed layout, per panel and per k-group g (4 k values):
//      Ap: P rows x 4 bytes   -> Ap[g*P*4 + i*4 + t] = A[(m0+i)*K + 4g + t]
//      Bp: P cols x 4 bytes   -> Bp[g*P*4 + j*4 + t] = B[(4g+t)*N + n0 + j]
//  The same layout is exactly what NEON SDOT wants, so the baseline reuses it.

size_t sme_s8_packed_a_bytes(int M, int K) { int P = s8_panel(); return (size_t)round_up(M, P) * (size_t)round_up(K, 4); }
size_t sme_s8_packed_b_bytes(int K, int N) { int P = s8_panel(); return (size_t)round_up(N, P) * (size_t)round_up(K, 4); }

void sme_s8_pack_a(const int8_t *A, int M, int K, int8_t *Ap) {
    const int P = s8_panel();
    const int KG = round_up(K, 4) / 4;
    for (int m0 = 0; m0 < M; m0 += P) {
        int8_t *dst = Ap + (size_t)(m0 / P) * KG * P * 4;
        for (int g = 0; g < KG; g++) {
            for (int i = 0; i < P; i++) {
                for (int t = 0; t < 4; t++) {
                    const int r = m0 + i, k = 4 * g + t;
                    dst[(size_t)g * P * 4 + i * 4 + t] = (r < M && k < K) ? A[(size_t)r * K + k] : 0;
                }
            }
        }
    }
}

void sme_s8_pack_b(const int8_t *B, int K, int N, int8_t *Bp) {
    const int P = s8_panel();
    const int KG = round_up(K, 4) / 4;
    for (int n0 = 0; n0 < N; n0 += P) {
        int8_t *dst = Bp + (size_t)(n0 / P) * KG * P * 4;
        for (int g = 0; g < KG; g++) {
            for (int j = 0; j < P; j++) {
                for (int t = 0; t < 4; t++) {
                    const int c = n0 + j, k = 4 * g + t;
                    dst[(size_t)g * P * 4 + j * 4 + t] = (c < N && k < K) ? B[(size_t)k * N + c] : 0;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// INT8 SME2 kernel
// ---------------------------------------------------------------------------------------

SME_TARGET __arm_new("za") __arm_locally_streaming
static void sme_s8_kernel(int M, int N, int K,
                          const int8_t *Ap, const int8_t *Bp,
                          int32_t *C, int ldc) {
    const int64_t vlw = (int64_t)svcntw();  // int32 lanes (16)
    const int64_t vlb = (int64_t)svcntb();  // bytes per vector (64)
    const int64_t P   = 2 * vlw;
    const int64_t KG  = (K + 3) / 4;
    const svbool_t  pt8 = svptrue_b8();
    const svcount_t pc8 = svptrue_c8();

    for (int64_t m0 = 0; m0 < M; m0 += P) {
        const int8_t *a_panel = Ap + (size_t)(m0 / P) * KG * P * 4;
        for (int64_t n0 = 0; n0 < N; n0 += P) {
            const int8_t *a = a_panel;
            const int8_t *b = Bp + (size_t)(n0 / P) * KG * P * 4;

            svzero_za();

            for (int64_t g = 0; g < KG; g++) {
                svint8x2_t av = svld1_s8_x2(pc8, a);
                svint8x2_t bv = svld1_s8_x2(pc8, b);
                svint8_t a0 = svget2(av, 0), a1 = svget2(av, 1);
                svint8_t b0 = svget2(bv, 0), b1 = svget2(bv, 1);
                svmopa_za32_s8_m(0, pt8, pt8, a0, b0);
                svmopa_za32_s8_m(1, pt8, pt8, a0, b1);
                svmopa_za32_s8_m(2, pt8, pt8, a1, b0);
                svmopa_za32_s8_m(3, pt8, pt8, a1, b1);
                a += 2 * vlb;
                b += 2 * vlb;
            }

            const svbool_t pn0 = svwhilelt_b32_s64(n0, (int64_t)N);
            const svbool_t pn1 = svwhilelt_b32_s64(n0 + vlw, (int64_t)N);
            for (int64_t i = 0; i < vlw; i++) {
                const int64_t r0 = m0 + i, r1 = m0 + vlw + i;
                if (r0 < M) {
                    int32_t *dst = C + (size_t)r0 * ldc + n0;
                    svst1_hor_za32(0, (uint32_t)i, pn0, dst);
                    svst1_hor_za32(1, (uint32_t)i, pn1, dst + vlw);
                }
                if (r1 < M) {
                    int32_t *dst = C + (size_t)r1 * ldc + n0;
                    svst1_hor_za32(2, (uint32_t)i, pn0, dst);
                    svst1_hor_za32(3, (uint32_t)i, pn1, dst + vlw);
                }
            }
        }
    }
}

void sme_s8_gemm_packed(int M, int N, int K,
                        const int8_t *Ap, const int8_t *Bp,
                        int32_t *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0 || !sme_cpu_has_sme2()) return;
    sme_s8_kernel(M, N, K, Ap, Bp, C, ldc);
}

// ---------------------------------------------------------------------------------------
// INT8 NEON baseline (SDOT), 8x8 micro-kernel on the same packed layout
// ---------------------------------------------------------------------------------------

#define SDOT_ROW(R, AV, LANE)                                     \
    acc[R][0] = vdotq_laneq_s32(acc[R][0], b0, AV, LANE);         \
    acc[R][1] = vdotq_laneq_s32(acc[R][1], b1, AV, LANE);

__attribute__((target("dotprod")))
void neon_s8_gemm_packed(int M, int N, int K,
                         const int8_t *Ap, const int8_t *Bp,
                         int32_t *C, int ldc) {
    const int P = s8_panel();
    const int KG = round_up(K, 4) / 4;
    for (int m0 = 0; m0 < M; m0 += P) {
        const int8_t *a_panel = Ap + (size_t)(m0 / P) * KG * P * 4;
        for (int n0 = 0; n0 < N; n0 += P) {
            const int8_t *b_panel = Bp + (size_t)(n0 / P) * KG * P * 4;
            for (int r0 = 0; r0 < P && m0 + r0 < M; r0 += 8) {
                for (int c0 = 0; c0 < P && n0 + c0 < N; c0 += 8) {
                    int32x4_t acc[8][2];
                    for (int r = 0; r < 8; r++) { acc[r][0] = vdupq_n_s32(0); acc[r][1] = vdupq_n_s32(0); }
                    const int8_t *a = a_panel + r0 * 4;
                    const int8_t *b = b_panel + c0 * 4;
                    for (int g = 0; g < KG; g++) {
                        const int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);  // rows r0..r0+3, r0+4..r0+7
                        const int8x16_t b0 = vld1q_s8(b), b1 = vld1q_s8(b + 16);  // cols c0..c0+3, c0+4..c0+7
                        SDOT_ROW(0, a0, 0) SDOT_ROW(1, a0, 1) SDOT_ROW(2, a0, 2) SDOT_ROW(3, a0, 3)
                        SDOT_ROW(4, a1, 0) SDOT_ROW(5, a1, 1) SDOT_ROW(6, a1, 2) SDOT_ROW(7, a1, 3)
                        a += P * 4;
                        b += P * 4;
                    }
                    int32_t tmp[8][8];
                    for (int r = 0; r < 8; r++) { vst1q_s32(tmp[r], acc[r][0]); vst1q_s32(tmp[r] + 4, acc[r][1]); }
                    const int rows = (M - m0 - r0) < 8 ? (M - m0 - r0) : 8;
                    const int cols = (N - n0 - c0) < 8 ? (N - n0 - c0) : 8;
                    for (int r = 0; r < rows; r++)
                        memcpy(C + (size_t)(m0 + r0 + r) * ldc + n0 + c0, tmp[r], sizeof(int32_t) * cols);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Scalar references
// ---------------------------------------------------------------------------------------

void ref_f32_gemm(int M, int N, int K, const float *A, const float *B,
                  const float *bias, int relu, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double s = bias ? bias[j] : 0.0;
            for (int k = 0; k < K; k++) s += (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = (relu && s < 0) ? 0.0f : (float)s;
        }
    }
}

void ref_s8_gemm(int M, int N, int K, const int8_t *A, const int8_t *B, int32_t *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = s;
        }
    }
}
