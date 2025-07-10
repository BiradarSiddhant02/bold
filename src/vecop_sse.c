#include "vecop_sse.h"

__attribute__((target("sse2")))
double euclidean_distance_f64_sse(const double* vec_a, const double* vec_b, const size_t length) {
    __m128d sum = _mm_setzero_pd();
    size_t i = 0;

    for (; i + 2 <= length; i += 2) {
        __m128d va = _mm_loadu_pd(vec_a + i);
        __m128d vb = _mm_loadu_pd(vec_b + i);
        __m128d diff = _mm_sub_pd(va, vb);
        __m128d sq = _mm_mul_pd(diff, diff);
        sum = _mm_add_pd(sum, sq);
    }

    double buffer[2];
    _mm_storeu_pd(buffer, sum);
    double distance = buffer[0] + buffer[1];

    for (; i < length; i++) {
        double diff = vec_a[i] - vec_b[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

__attribute__((target("sse")))
float euclidean_distance_f32_sse(const float* vec_a, const float* vec_b, const size_t length) {
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;

    for (; i + 4 <= length; i += 4) {
        __m128 va = _mm_loadu_ps(vec_a + i);
        __m128 vb = _mm_loadu_ps(vec_b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }

    float buffer[4];
    _mm_storeu_ps(buffer, sum);
    float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3];

    for (; i < length; i++) {
        float diff = vec_a[i] - vec_b[i];
        distance += diff * diff;
    }

    return sqrtf(distance);
}

__attribute__((target("sse2")))
double* batched_euclidean_f64_sse(const double** vecs, const double* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    double* distances = (double*)malloc(sizeof(double) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = euclidean_distance_f64_sse(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("sse")))
float* batched_euclidean_f32_sse(const float** vecs, const float* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    float* distances = (float*)malloc(sizeof(float) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = euclidean_distance_f32_sse(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("sse2")))
double* centroid_f64_sse(const double** vecs, const size_t n_vectors, const size_t length) {
    double* centroid = (double*)malloc(sizeof(double) * length);

    for (size_t k = 0; k < length; k++) centroid[k] = 0.0;

    size_t i, j;

    for (j = 0; j + 2 <= length; j += 2) {
        __m128d sum = _mm_setzero_pd();

        for (i = 0; i < n_vectors; i++) {
            __m128d v = _mm_loadu_pd(&vecs[i][j]);
            sum = _mm_add_pd(sum, v);
        }

        __m128d avg = _mm_div_pd(sum, _mm_set1_pd((double)n_vectors));
        _mm_storeu_pd(&centroid[j], avg);
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

__attribute__((target("sse")))
float* centroid_f32_sse(const float** vecs, const size_t n_vectors, const size_t length) {
    float* centroid = (float*)malloc(sizeof(float) * length);

    for (size_t k = 0; k < length; k++) centroid[k] = 0.0f;

    size_t i, j;

    for (j = 0; j + 4 <= length; j += 4) {
        __m128 sum = _mm_setzero_ps();

        for (i = 0; i < n_vectors; i++) {
            __m128 v = _mm_loadu_ps(&vecs[i][j]);
            sum = _mm_add_ps(sum, v);
        }

        __m128 avg = _mm_div_ps(sum, _mm_set1_ps((float)n_vectors));
        _mm_storeu_ps(&centroid[j], avg);
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

__attribute__((target("sse2")))
float manhattan_distance_f32_sse(const float* vec_a, const float* vec_b, const size_t length) {
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;

    // Sign bit mask for floats: 0x7FFFFFFF
    const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

    for (; i + 4 <= length; i += 4) {
        __m128 va = _mm_loadu_ps(vec_a + i);
        __m128 vb = _mm_loadu_ps(vec_b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        __m128 abs_diff = _mm_and_ps(diff, sign_mask);
        sum = _mm_add_ps(sum, abs_diff);
    }

    // Horizontal sum of __m128
    float buffer[4];
    _mm_storeu_ps(buffer, sum);
    float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3];

    // Handle remainder elements
    for (; i < length; ++i) {
        distance += fabsf(vec_a[i] - vec_b[i]);
    }

    return distance;
}

__attribute__((target("sse2")))
double manhattan_distance_f64_sse(const double* vec_a, const double* vec_b, const size_t length) {
    __m128d sum = _mm_setzero_pd();
    size_t i = 0;

    // Mask to clear sign bit: 0x7FFFFFFFFFFFFFFF
    const __m128d sign_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    for (; i + 2 <= length; i += 2) {
        __m128d va = _mm_loadu_pd(vec_a + i);
        __m128d vb = _mm_loadu_pd(vec_b + i);
        __m128d diff = _mm_sub_pd(va, vb);
        __m128d abs_diff = _mm_and_pd(diff, sign_mask);
        sum = _mm_add_pd(sum, abs_diff);
    }

    // Horizontal sum of __m128d
    double buffer[2];
    _mm_storeu_pd(buffer, sum);
    double distance = buffer[0] + buffer[1];

    // Handle remainder elements
    for (; i < length; ++i) {
        distance += fabs(vec_a[i] - vec_b[i]);
    }

    return distance;
}

__attribute__((target("sse2")))
double* batched_manhattan_f64_sse(const double** vecs, const double* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    double* distances = (double*)malloc(sizeof(double) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = manhattan_distance_f64_sse(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("sse2")))
float* batched_manhattan_f32_sse(const float** vecs, const float* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    float* distances = (float*)malloc(sizeof(float) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = manhattan_distance_f32_sse(vecs[i], vec, length);
        }
    }

    return distances;
}