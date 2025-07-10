#ifndef VECOP_SSE_H
#define VECOP_SSE_H

#include <math.h>
#include <xmmintrin.h>   // SSE
#include <emmintrin.h>   // SSE2 (for double-precision)
#include <omp.h>
#include <stdlib.h>

__attribute__((target("sse2")))
double euclidean_distance_f64_sse(const double*, const double*, const size_t);
__attribute__((target("sse2")))
float euclidean_distance_f32_sse(const float*, const float*, const size_t);

__attribute__((target("sse2")))
double* batched_euclidean_f64_sse(const double**, const double*, const size_t, const size_t, size_t);
__attribute__((target("sse2")))
float* batched_euclidean_f32_sse(const float**, const float*, const size_t, const size_t, size_t);

__attribute__((target("sse2")))
double* centroid_f64_sse(const double**, const size_t, const size_t);
__attribute__((target("sse2")))
float* centroid_f32_sse(const float**, const size_t, const size_t);

__attribute__((target("sse2")))
double manhattan_distance_f64_sse(const double*, const double*, const size_t);
__attribute__((target("sse2")))
float manhattan_distance_f32_sse(const float*, const float*, const size_t);

__attribute__((target("sse2")))
double* batched_manhattan_f64_sse(const double**, const double*, const size_t, const size_t, size_t);
__attribute__((target("sse2")))
float* batched_manhattan_f32_sse(const float**, const float*, const size_t, const size_t, size_t);

#endif  // VECOP_SSE_H
