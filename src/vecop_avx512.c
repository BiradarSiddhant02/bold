#include "vecop_avx512.h"

__attribute__((target("avx512f")))
double euclidean_distance_f64_avx512(const double* vec_a, const double* vec_b, const size_t length) {
    __m512d sum = _mm512_setzero_pd();
    size_t i = 0;

    for (; i + 8 <= length; i += 8) {
        __m512d va = _mm512_loadu_pd(vec_a + i);
        __m512d vb = _mm512_loadu_pd(vec_b + i);
        __m512d diff = _mm512_sub_pd(va, vb);
        __m512d sq = _mm512_mul_pd(diff, diff);
        sum = _mm512_add_pd(sum, sq);
    }

    double buffer[8];
    _mm512_storeu_pd(buffer, sum);
    double distance = 0.0;
    for (int j = 0; j < 8; j++) distance += buffer[j];

    for (; i < length; i++) {
        double diff = vec_a[i] - vec_b[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

__attribute__((target("avx512f")))
float euclidean_distance_f32_avx512(const float* vec_a, const float* vec_b, const size_t length) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= length; i += 16) {
        __m512 va = _mm512_loadu_ps(vec_a + i);
        __m512 vb = _mm512_loadu_ps(vec_b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        __m512 sq = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, sq);
    }

    float buffer[16];
    _mm512_storeu_ps(buffer, sum);
    float distance = 0.0f;
    for (int j = 0; j < 16; j++) distance += buffer[j];

    for (; i < length; i++) {
        float diff = vec_a[i] - vec_b[i];
        distance += diff * diff;
    }

    return sqrtf(distance);
}

__attribute__((target("avx512f")))
double* batched_euclidean_f64_avx512(const double** vecs, const double* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    double* distances = (double*)malloc(sizeof(double) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = euclidean_distance_f64_avx512(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("avx512f")))
float* batched_euclidean_f32_avx512(const float** vecs, const float* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    float* distances = (float*)malloc(sizeof(float) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = euclidean_distance_f32_avx512(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("avx512f")))
double* centroid_f64_avx512(const double** vecs, const size_t n_vectors, const size_t length) {
    double* centroid = (double*)malloc(sizeof(double) * length);
    for (size_t k = 0; k < length; k++) centroid[k] = 0.0;

    size_t i, j;

    for (j = 0; j + 8 <= length; j += 8) {
        __m512d sum = _mm512_setzero_pd();

        for (i = 0; i < n_vectors; i++) {
            __m512d v = _mm512_loadu_pd(&vecs[i][j]);
            sum = _mm512_add_pd(sum, v);
        }

        __m512d avg = _mm512_div_pd(sum, _mm512_set1_pd((double)n_vectors));
        _mm512_storeu_pd(&centroid[j], avg);
    }

    for (; j < length; j++) {
        double sum = 0.0;
        for (i = 0; i < n_vectors; i++) {
            sum += vecs[i][j];
        }
        centroid[j] = sum / (double)n_vectors;
    }

    return centroid;
}

__attribute__((target("avx512f")))
float* centroid_f32_avx512(const float** vecs, const size_t n_vectors, const size_t length) {
    float* centroid = (float*)malloc(sizeof(float) * length);
    for (size_t k = 0; k < length; k++) centroid[k] = 0.0f;

    size_t i, j;

    for (j = 0; j + 16 <= length; j += 16) {
        __m512 sum = _mm512_setzero_ps();

        for (i = 0; i < n_vectors; i++) {
            __m512 v = _mm512_loadu_ps(&vecs[i][j]);
            sum = _mm512_add_ps(sum, v);
        }

        __m512 avg = _mm512_div_ps(sum, _mm512_set1_ps((float)n_vectors));
        _mm512_storeu_ps(&centroid[j], avg);
    }

    for (; j < length; j++) {
        float sum = 0.0f;
        for (i = 0; i < n_vectors; i++) {
            sum += vecs[i][j];
        }
        centroid[j] = sum / (float)n_vectors;
    }

    return centroid;
}

__attribute__((target("avx512f,avx512dq")))
double manhattan_distance_f64_avx512(const double* vec_a, const double* vec_b, const size_t length) {
    __m512d sum = _mm512_setzero_pd();
    size_t i = 0;

    // Mask to clear sign bit: 0x7FFFFFFFFFFFFFFF
    const __m512d sign_mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));

    for (; i + 8 <= length; i += 8) {
        __m512d va = _mm512_loadu_pd(vec_a + i);
        __m512d vb = _mm512_loadu_pd(vec_b + i);
        __m512d diff = _mm512_sub_pd(va, vb);
        __m512d abs_diff = _mm512_and_pd(diff, sign_mask);
        sum = _mm512_add_pd(sum, abs_diff);
    }

    // Horizontal sum of 8 doubles
    double buffer[8];
    _mm512_storeu_pd(buffer, sum);
    double distance = 0.0;
    for (int j = 0; j < 8; ++j) {
        distance += buffer[j];
    }

    // Handle tail elements
    for (; i < length; ++i) {
        distance += fabs(vec_a[i] - vec_b[i]);
    }

    return distance;
}

__attribute__((target("avx512f,avx512dq")))
float manhattan_distance_f32_avx512(const float* vec_a, const float* vec_b, const size_t length) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;

    // Mask to clear sign bit: 0x7FFFFFFF
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

    for (; i + 16 <= length; i += 16) {
        __m512 va = _mm512_loadu_ps(vec_a + i);
        __m512 vb = _mm512_loadu_ps(vec_b + i);
        __m512 diff = _mm512_sub_ps(va, vb);
        __m512 abs_diff = _mm512_and_ps(diff, sign_mask);
        sum = _mm512_add_ps(sum, abs_diff);
    }

    // Horizontal sum of 16 floats
    float buffer[16];
    _mm512_storeu_ps(buffer, sum);
    float distance = 0.0f;
    for (int j = 0; j < 16; ++j) {
        distance += buffer[j];
    }

    // Handle tail elements
    for (; i < length; ++i) {
        distance += fabsf(vec_a[i] - vec_b[i]);
    }

    return distance;
}

__attribute__((target("avx512f,avx512dq")))
double* batched_manhattan_f64_avx512(const double** vecs, const double* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    double* distances = (double*)malloc(sizeof(double) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = manhattan_distance_f64_avx512(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("avx512f,avx512dq")))
float* batched_manhattan_f32_avx512(const float** vecs, const float* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    float* distances = (float*)malloc(sizeof(float) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = manhattan_distance_f32_avx512(vecs[i], vec, length);
        }
    }

    return distances;
}