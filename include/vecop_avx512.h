#ifndef VECOP_AVX512_H
#define VECOP_AVX512_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

__attribute__((target("avx512f")))
double euclidean_distance_f64_avx512(const double*, const double*, const size_t);
__attribute__((target("avx512f")))
float euclidean_distance_f32_avx512(const float*, const float*, const size_t);

__attribute__((target("avx512f")))
double* batched_euclidean_f64_avx512(const double**, const double*, const size_t, const size_t, size_t);
__attribute__((target("avx512f")))
float* batched_euclidean_f32_avx512(const float**, const float*, const size_t, const size_t, size_t);

__attribute__((target("avx512f")))
double* centroid_f64_avx512(const double**, const size_t, const size_t);
__attribute__((target("avx512f")))
float* centroid_f32_avx512(const float**, const size_t, const size_t);

__attribute__((target("avx512f,avx512dq")))
double manhattan_distance_f64_avx512(const double*, const double*, const size_t);
__attribute__((target("avx512f,avx512dq")))
float manhattan_distance_f32_avx512(const float*, const float*, const size_t);

__attribute__((target("avx512f")))
double* batched_manhattan_f64_avx512(const double**, const double*, const size_t, const size_t, size_t);
__attribute__((target("avx512f")))
float* batched_manhattan_f32_avx512(const float**, const float*, const size_t, const size_t, size_t);

#endif  // VECOP_AVX512_H
