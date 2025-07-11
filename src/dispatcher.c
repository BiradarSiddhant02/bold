#define _GNU_SOURCE
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "dispatcher.h"
#include "vecop_avx512.h"
#include "vecop_avx2.h"
#include "vecop_avx.h"
#include "vecop_sse.h"

// CPU feature flags and dispatch table
static struct {
    bool avx512;
    bool avx2;
    bool avx;
    bool sse;
    bool initialized;
    char force_arch[16]; // New: Store forced architecture

    // Euclidean function pointers
    double (*e_dist_f64)(const double*, const double*, size_t);
    float  (*e_dist_f32)(const float*, const float*, size_t);
    double* (*e_batch_f64)(const double**, const double*, size_t, size_t, size_t);
    float*  (*e_batch_f32)(const float**, const float*, size_t, size_t, size_t);

    // Centroid function pointers
    double* (*centroid_f64)(const double**, size_t, size_t);
    float*  (*centroid_f32)(const float**, size_t, size_t);

    // Manhattan function pointers
    double (*m_dist_f64)(const double*, const double*, size_t);
    float  (*m_dist_f32)(const float*, const float*, size_t);
    double* (*m_batch_f64)(const double**, const double*, size_t, size_t, size_t);
    float*  (*m_batch_f32)(const float**, const float*, size_t, size_t, size_t);
} cpu_caps = {0};

// Runtime SIMD detection and dispatch initialization
static void detect_and_init_dispatch() {
    if (cpu_caps.initialized) return;

    cpu_caps.avx512 = __builtin_cpu_supports("avx512f");
    cpu_caps.avx2   = __builtin_cpu_supports("avx2");
    cpu_caps.avx    = __builtin_cpu_supports("avx");
    cpu_caps.sse    = __builtin_cpu_supports("sse2");

    // Priority: Forced Arch > AVX-512 > AVX2 > AVX > SSE > Scalar
    if (strlen(cpu_caps.force_arch) > 0) {
        // Reset all flags first, then enable only the requested architecture
        cpu_caps.avx512 = false;
        cpu_caps.avx2 = false;
        cpu_caps.avx = false;
        cpu_caps.sse = false;
        
        if (strcmp(cpu_caps.force_arch, "AVX512") == 0) {
            cpu_caps.avx512 = true;  // Force enable, assume CPU supports it
        } else if (strcmp(cpu_caps.force_arch, "AVX2") == 0) {
            cpu_caps.avx2 = true;    // Force enable, assume CPU supports it
        } else if (strcmp(cpu_caps.force_arch, "AVX") == 0) {
            cpu_caps.avx = true;     // Force enable, assume CPU supports it
        } else if (strcmp(cpu_caps.force_arch, "SSE") == 0) {
            cpu_caps.sse = true;     // Force enable, assume CPU supports it
        }
        // "SCALAR" will result in all flags being false, using the fallback
    }

    // Priority: AVX-512 > AVX2 > AVX > SSE > Scalar fallback
    if (cpu_caps.avx512) {
        cpu_caps.e_dist_f64 = euclidean_distance_f64_avx512;
        cpu_caps.e_dist_f32 = euclidean_distance_f32_avx512;
        cpu_caps.e_batch_f64 = batched_euclidean_f64_avx512;
        cpu_caps.e_batch_f32 = batched_euclidean_f32_avx512;
        cpu_caps.centroid_f64 = centroid_f64_avx512;
        cpu_caps.centroid_f32 = centroid_f32_avx512;
        cpu_caps.m_dist_f64 = manhattan_distance_f64_avx512;
        cpu_caps.m_dist_f32 = manhattan_distance_f32_avx512;
        cpu_caps.m_batch_f64 = batched_manhattan_f64_avx512;
        cpu_caps.m_batch_f32 = batched_manhattan_f32_avx512;
    } else if (cpu_caps.avx2) {
        cpu_caps.e_dist_f64 = euclidean_distance_f64_avx2;
        cpu_caps.e_dist_f32 = euclidean_distance_f32_avx2;
        cpu_caps.e_batch_f64 = batched_euclidean_f64_avx2;
        cpu_caps.e_batch_f32 = batched_euclidean_f32_avx2;
        cpu_caps.centroid_f64 = centroid_f64_avx2;
        cpu_caps.centroid_f32 = centroid_f32_avx2;
        cpu_caps.m_dist_f64 = manhattan_distance_f64_avx2;
        cpu_caps.m_dist_f32 = manhattan_distance_f32_avx2;
        cpu_caps.m_batch_f64 = batched_manhattan_f64_avx2;
        cpu_caps.m_batch_f32 = batched_manhattan_f32_avx2;
    } else if (cpu_caps.avx) {
        cpu_caps.e_dist_f64 = euclidean_distance_f64_avx;
        cpu_caps.e_dist_f32 = euclidean_distance_f32_avx;
        cpu_caps.e_batch_f64 = batched_euclidean_f64_avx;
        cpu_caps.e_batch_f32 = batched_euclidean_f32_avx;
        cpu_caps.centroid_f64 = centroid_f64_avx;
        cpu_caps.centroid_f32 = centroid_f32_avx;
        cpu_caps.m_dist_f64 = manhattan_distance_f64_avx;
        cpu_caps.m_dist_f32 = manhattan_distance_f32_avx;
        cpu_caps.m_batch_f64 = batched_manhattan_f64_avx;
        cpu_caps.m_batch_f32 = batched_manhattan_f32_avx;
    } else if (cpu_caps.sse) {
        cpu_caps.e_dist_f64 = euclidean_distance_f64_sse;
        cpu_caps.e_dist_f32 = euclidean_distance_f32_sse;
        cpu_caps.e_batch_f64 = batched_euclidean_f64_sse;
        cpu_caps.e_batch_f32 = batched_euclidean_f32_sse;
        cpu_caps.centroid_f64 = centroid_f64_sse;
        cpu_caps.centroid_f32 = centroid_f32_sse;
        cpu_caps.m_dist_f64 = manhattan_distance_f64_sse;
        cpu_caps.m_dist_f32 = manhattan_distance_f32_sse;
        cpu_caps.m_batch_f64 = batched_manhattan_f64_sse;
        cpu_caps.m_batch_f32 = batched_manhattan_f32_sse;
    } else {
        // scalar fallback (set to NULL to use fallback logic)
        cpu_caps.e_dist_f64 = NULL;
        cpu_caps.e_dist_f32 = NULL;
        cpu_caps.e_batch_f64 = NULL;
        cpu_caps.e_batch_f32 = NULL;
        cpu_caps.centroid_f64 = NULL;
        cpu_caps.centroid_f32 = NULL;
        cpu_caps.m_dist_f64 = NULL;
        cpu_caps.m_dist_f32 = NULL;
        cpu_caps.m_batch_f64 = NULL;
        cpu_caps.m_batch_f32 = NULL;
    }

    cpu_caps.initialized = true;
}

// Dispatcher: Euclidean Distance f64
double euclidean_distance_f64(const double* a, const double* b, size_t len) {
    detect_and_init_dispatch();
    if (cpu_caps.e_dist_f64) return cpu_caps.e_dist_f64(a, b, len);

    // Scalar fallback
    double dist = 0.0;
    for (register size_t i = 0; i < len; ++i) {
        double d = a[i] - b[i];
        dist += d * d;
    }
    return sqrt(dist);
}

// Dispatcher: Euclidean Distance f32
float euclidean_distance_f32(const float* a, const float* b, size_t len) {
    detect_and_init_dispatch();
    if (cpu_caps.e_dist_f32) return cpu_caps.e_dist_f32(a, b, len);

    // Scalar fallback
    float dist = 0.0f;
    for (register size_t i = 0; i < len; ++i) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return sqrtf(dist);
}

// Dispatcher: Batched Euclidean Distance f64
double* batched_euclidean_f64(const double** vecs, const double* vec, size_t n, size_t len, size_t num_threads) {
    detect_and_init_dispatch();
    if (cpu_caps.e_batch_f64) return cpu_caps.e_batch_f64(vecs, vec, n, len, num_threads);

    // Scalar fallback
    double* out = malloc(sizeof(double) * n);
    for (register size_t i = 0; i < n; ++i)
        out[i] = euclidean_distance_f64(vecs[i], vec, len);
    return out;
}

// Dispatcher: Batched Euclidean Distance f32
float* batched_euclidean_f32(const float** vecs, const float* vec, size_t n, size_t len, size_t num_threads) {
    detect_and_init_dispatch();
    if (cpu_caps.e_batch_f32) return cpu_caps.e_batch_f32(vecs, vec, n, len, num_threads);

    // Scalar fallback
    float* out = malloc(sizeof(float) * n);
    for (register size_t i = 0; i < n; ++i)
        out[i] = euclidean_distance_f32(vecs[i], vec, len);
    return out;
}

// Dispatcher: Centroid f64
double* centroid_f64(const double** vecs, size_t n, size_t len) {
    detect_and_init_dispatch();
    if (cpu_caps.centroid_f64) return cpu_caps.centroid_f64(vecs, n, len);

    // Scalar fallback
    double* c = calloc(len, sizeof(double));
    for (register size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < len; ++j)
            c[j] += vecs[i][j];
    for (size_t j = 0; j < len; ++j)
        c[j] /= (double)n;
    return c;
}

// Dispatcher: Centroid f32
float* centroid_f32(const float** vecs, size_t n, size_t len) {
    detect_and_init_dispatch();
    if (cpu_caps.centroid_f32) return cpu_caps.centroid_f32(vecs, n, len);

    // Scalar fallback
    float* c = calloc(len, sizeof(float));
    for (register size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < len; ++j)
            c[j] += vecs[i][j];
    for (size_t j = 0; j < len; ++j)
        c[j] /= (float)n;
    return c;
}

// Dispatcher: manhattam Distance f64
double manhattan_distance_f64(const double* a, const double* b, size_t len) {
    detect_and_init_dispatch();
    if (cpu_caps.m_dist_f64) return cpu_caps.m_dist_f64(a, b, len);

    // Scalar fallback
    double dist = 0.0;
    for (register size_t i = 0; i < len; ++i) {
        dist += fabs(a[i] - b[i]);
    }
    return dist;
}

// Dispatcher: manhattan Distance f32
float manhattan_distance_f32(const float* a, const float* b, size_t len) {
    detect_and_init_dispatch();
    if (cpu_caps.m_dist_f32) return cpu_caps.m_dist_f32(a, b, len);

    // Scalar fallback
    float dist = 0.0f;
    for (register size_t i = 0; i < len; ++i) {
        dist += fabsf(a[i] - b[i]);
    }
    return dist;
}

// Dispatcher: Batched manhattan Distance f64
double* batched_manhattan_f64(const double** vecs, const double* vec, size_t n, size_t len, size_t num_threads) {
    detect_and_init_dispatch();
    if (cpu_caps.m_batch_f64) return cpu_caps.m_batch_f64(vecs, vec, n, len, num_threads);

    // Scalar fallback
    double* out = malloc(sizeof(double) * n);
    for (register size_t i = 0; i < n; ++i)
        out[i] = manhattan_distance_f64(vecs[i], vec, len);
    return out;
}

// Dispatcher: Batched manhattan Distance f32
float* batched_manhattan_f32(const float** vecs, const float* vec, size_t n, size_t len, size_t num_threads) {
    detect_and_init_dispatch();
    if (cpu_caps.m_batch_f32) return cpu_caps.m_batch_f32(vecs, vec, n, len, num_threads);

    // Scalar fallback
    float* out = malloc(sizeof(float) * n);
    for (register size_t i = 0; i < n; ++i)
        out[i] = manhattan_distance_f32(vecs[i], vec, len);
    return out;
}

// === Architecture Info ===
const char* vecop_print_arch(void) {
    detect_and_init_dispatch();
    if (cpu_caps.avx512) {
        return "AVX512";
    } else if (cpu_caps.avx2) {
        return "AVX2";
    } else if (cpu_caps.avx) {
        return "AVX";
    } else if (cpu_caps.sse) {
        return "SSE";
    } else {
        return "SCALAR";
    }
}

VECOP_API void vecop_set_arch(const char* arch) {
    if (arch) {
        strncpy(cpu_caps.force_arch, arch, sizeof(cpu_caps.force_arch) - 1);
        cpu_caps.force_arch[sizeof(cpu_caps.force_arch) - 1] = '\0';
    } else {
        cpu_caps.force_arch[0] = '\0';
    }
    // Reset flags to allow re-initialization
    cpu_caps.initialized = false;
    cpu_caps.avx512 = false;
    cpu_caps.avx2 = false;
    cpu_caps.avx = false;
    cpu_caps.sse = false;
}