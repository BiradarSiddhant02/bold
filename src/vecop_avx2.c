#include "vecop_avx2.h"

__attribute__((target("avx2")))
double euclidean_distance_f64_avx2(const double* vec_a, const double* vec_b, const size_t length) {
    register __m256d sum = _mm256_setzero_pd();
    register size_t i = 0;

    for (; i + 4 <= length; i += 4) {
        __m256d va = _mm256_loadu_pd(vec_a + i);
        __m256d vb = _mm256_loadu_pd(vec_b + i);
        __m256d diff = _mm256_sub_pd(va, vb);
        __m256d sq = _mm256_mul_pd(diff, diff);
        sum = _mm256_add_pd(sum, sq);
    }

    double buffer[4];
    _mm256_storeu_pd(buffer, sum);
    double distance = buffer[0] + buffer[1] + buffer[2] + buffer[3];

    for (; i < length; i++) {
        double diff = vec_a[i] - vec_b[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

__attribute__((target("avx2")))
float euclidean_distance_f32_avx2(const float* vec_a, const float* vec_b, const size_t length) {
    register __m256 sum = _mm256_setzero_ps();
    register size_t i = 0;

    for (; i + 8 <= length; i += 8) {
        __m256 va = _mm256_loadu_ps(vec_a + i);
        __m256 vb = _mm256_loadu_ps(vec_b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                     buffer[4] + buffer[5] + buffer[6] + buffer[7];

    for (; i < length; i++) {
        float diff = vec_a[i] - vec_b[i];
        distance += diff * diff;
    }

    return sqrtf(distance);
}

__attribute__((target("avx2")))
double* batched_euclidean_f64_avx2(const double** vecs, const double* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    /**
     * 
     */

    double* distances = (double*)malloc(sizeof(double) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (register size_t i = 0; i < n_vectors; i++) { distances[i] = euclidean_distance_f64_avx2(vecs[i], vec, length); }
    }

    return distances;
}

__attribute__((target("avx2")))
float* batched_euclidean_f32_avx2(const float** vecs, const float* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    /**
     * 
     */

    float* distances = (float*)malloc(sizeof(float) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (register size_t i = 0; i < n_vectors; i++) { distances[i] = euclidean_distance_f32_avx2(vecs[i], vec, length); }
    }

    return distances;
}

__attribute__((target("avx2")))
double* centroid_f64_avx2(const double** vecs, const size_t n_vectors, const size_t length) {
    double* centroid = (double*)malloc(sizeof(double) * length);
    
    // Initialize to zero
    for (register size_t k = 0; k < length; k++) {
        centroid[k] = 0.0;
    }

    register size_t i, j;

    // SIMD: process 4 doubles at a time
    for (j = 0; j + 4 <= length; j += 4) {
        register __m256d sum = _mm256_setzero_pd();

        for (i = 0; i < n_vectors; i++) {
            __m256d v = _mm256_loadu_pd(&vecs[i][j]);
            sum = _mm256_add_pd(sum, v);
        }

        // Divide by number of vectors
        __m256d avg = _mm256_div_pd(sum, _mm256_set1_pd((double)n_vectors));
        _mm256_storeu_pd(&centroid[j], avg);
    }

    // Handle remaining dimensions
    for (; j < length; j++) {
        double sum = 0.0;
        for (i = 0; i < n_vectors; i++) {
            sum += vecs[i][j];
        }
        centroid[j] = sum / (double)n_vectors;
    }

    return centroid;
}

__attribute__((target("avx2")))
float* centroid_f32_avx2(const float** vecs, const size_t n_vectors, const size_t length) {
    float* centroid = (float*)malloc(sizeof(float) * length);
    
    // Initialize to zero
    for (register size_t k = 0; k < length; k++) {
        centroid[k] = 0.0f;
    }

    register size_t i, j;

    // SIMD: process 8 floats at a time
    for (j = 0; j + 8 <= length; j += 8) {
        register __m256 sum = _mm256_setzero_ps();

        for (i = 0; i < n_vectors; i++) {
            __m256 v = _mm256_loadu_ps(&vecs[i][j]);
            sum = _mm256_add_ps(sum, v);
        }

        __m256 avg = _mm256_div_ps(sum, _mm256_set1_ps((float)n_vectors));
        _mm256_storeu_ps(&centroid[j], avg);
    }

    // Handle leftover dimensions (length not divisible by 8)
    for (; j < length; j++) {
        float sum = 0.0f;
        for (i = 0; i < n_vectors; i++) {
            sum += vecs[i][j];
        }
        centroid[j] = sum / (float)n_vectors;
    }

    return centroid;
}

__attribute__((target("avx2")))
float manhattan_distance_f32_avx2(const float* vec_a, const float* vec_b, const size_t length) {
    register __m256 sum = _mm256_setzero_ps();
    register size_t i = 0;

    for (; i + 8 <= length; i += 8) {
        __m256 va = _mm256_loadu_ps(vec_a + i);
        __m256 vb = _mm256_loadu_ps(vec_b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 neg_diff = _mm256_sub_ps(_mm256_setzero_ps(), diff);
        __m256 abs_diff = _mm256_max_ps(diff, neg_diff);  // |diff|
        sum = _mm256_add_ps(sum, abs_diff);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                     buffer[4] + buffer[5] + buffer[6] + buffer[7];

    for (; i < length; ++i)
        distance += fabsf(vec_a[i] - vec_b[i]);

    return distance;
}

__attribute__((target("avx2")))
double manhattan_distance_f64_avx2(const double* vec_a, const double* vec_b, const size_t length) {
    register __m256d sum = _mm256_setzero_pd();
    register size_t i = 0;

    for (; i + 4 <= length; i += 4) {
        __m256d va = _mm256_loadu_pd(vec_a + i);
        __m256d vb = _mm256_loadu_pd(vec_b + i);
        __m256d diff = _mm256_sub_pd(va, vb);
        __m256d neg_diff = _mm256_sub_pd(_mm256_setzero_pd(), diff);
        __m256d abs_diff = _mm256_max_pd(diff, neg_diff);  // |diff|
        sum = _mm256_add_pd(sum, abs_diff);
    }

    double buffer[4];
    _mm256_storeu_pd(buffer, sum);
    double distance = buffer[0] + buffer[1] + buffer[2] + buffer[3];

    for (; i < length; ++i)
        distance += fabs(vec_a[i] - vec_b[i]);

    return distance;
}


__attribute__((target("avx2")))
double* batched_manhattan_f64_avx2(const double** vecs, const double* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    double* distances = (double*)malloc(sizeof(double) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (register size_t i = 0; i < n_vectors; i++) {
            distances[i] = manhattan_distance_f64_avx2(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("avx2")))
float* batched_manhattan_f32_avx2(const float** vecs, const float* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    float* distances = (float*)malloc(sizeof(float) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (register size_t i = 0; i < n_vectors; i++) {
            distances[i] = manhattan_distance_f32_avx2(vecs[i], vec, length);
        }
    }

    return distances;
}