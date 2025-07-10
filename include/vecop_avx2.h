#ifndef VECOP_AVX2_H
#define VECOP_AVX2_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

__attribute__((target("avx2")))
double euclidean_distance_f64_avx2(const double*, const double*, const size_t);
__attribute__((target("avx2")))
float euclidean_distance_f32_avx2(const float*, const float*, const size_t);

__attribute__((target("avx2")))
double* batched_euclidean_f64_avx2(const double**, const double*, const size_t, const size_t, size_t);
__attribute__((target("avx2")))
float* batched_euclidean_f32_avx2(const float**, const float*, const size_t, const size_t, size_t);

__attribute__((target("avx2")))
double* centroid_f64_avx2(const double**, const size_t, const size_t);
__attribute__((target("avx2")))
float* centroid_f32_avx2(const float**, const size_t, const size_t);

__attribute__((target("avx2")))
double manhattan_distance_f64_avx2(const double*, const double*, const size_t);
__attribute__((target("avx2")))
float manhattan_distance_f32_avx2(const float*, const float*, const size_t);

__attribute__((target("avx2")))
double* batched_manhattan_f64_avx2(const double**, const double*, const size_t, const size_t, size_t);
__attribute__((target("avx2")))
float* batched_manhattan_f32_avx2(const float**, const float*, const size_t, const size_t, size_t);

#endif  // VECOP_AVX2_H