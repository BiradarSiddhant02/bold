#ifndef VECOP_AVX_H
#define VECOP_AVX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

__attribute__((target("avx")))
double euclidean_distance_f64_avx(const double*, const double*, const size_t);
__attribute__((target("avx")))
float euclidean_distance_f32_avx(const float*, const float*, const size_t);

__attribute__((target("avx")))
double* batched_euclidean_f64_avx(const double**, const double*, const size_t, const size_t, size_t);
__attribute__((target("avx")))
float* batched_euclidean_f32_avx(const float**, const float*, const size_t, const size_t, size_t);

__attribute__((target("avx")))
double* centroid_f64_avx(const double**, const size_t, const size_t);
__attribute__((target("avx")))
float* centroid_f32_avx(const float**, const size_t, const size_t);

__attribute__((target("avx")))
double manhattan_distance_f64_avx(const double*, const double*, const size_t);
__attribute__((target("avx")))
float manhattan_distance_f32_avx(const float*, const float*, const size_t);

__attribute__((target("avx")))
double* batched_manhattan_f64_avx(const double**, const double*, const size_t, const size_t, size_t);
__attribute__((target("avx")))
float* batched_manhattan_f32_avx(const float**, const float*, const size_t, const size_t, size_t);

#endif  // VECOP_AVX_H