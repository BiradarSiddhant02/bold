#include "vecop_avx.h"

__attribute__((target("avx")))
double euclidean_distance_f64_avx(const double* vec_a, const double* vec_b, size_t length) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 4 <= length; i += 4) {
        __m256d va = _mm256_loadu_pd(vec_a + i);
        __m256d vb = _mm256_loadu_pd(vec_b + i);
        __m256d diff = _mm256_sub_pd(va, vb);
        __m256d sq = _mm256_mul_pd(diff, diff);
        sum = _mm256_add_pd(sum, sq);
    }

    // Horizontal sum of __m256d
    __m128d vlow  = _mm256_castpd256_pd128(sum);
    __m128d vhigh = _mm256_extractf128_pd(sum, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // add the two 128-bit vectors
    __m128d hsum  = _mm_hadd_pd(vlow, vlow);
    double distance = _mm_cvtsd_f64(hsum);

    for (; i < length; i++) {
        double diff = vec_a[i] - vec_b[i];
        distance += diff * diff;
    }

    return sqrt(distance);
}

__attribute__((target("avx")))
float euclidean_distance_f32_avx(const float* vec_a, const float* vec_b, size_t length) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= length; i += 8) {
        __m256 va = _mm256_loadu_ps(vec_a + i);
        __m256 vb = _mm256_loadu_ps(vec_b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float distance = 0.0f;
    for (int j = 0; j < 8; j++) distance += buffer[j];

    for (; i < length; i++) {
        float diff = vec_a[i] - vec_b[i];
        distance += diff * diff;
    }

    return sqrtf(distance);
}

__attribute__((target("avx")))
double* batched_euclidean_f64_avx(const double** vecs, const double* vec, size_t n_vectors, size_t length, size_t num_threads) {
    double* distances = (double*)malloc(sizeof(double) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = euclidean_distance_f64_avx(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("avx")))
float* batched_euclidean_f32_avx(const float** vecs, const float* vec, size_t n_vectors, size_t length, size_t num_threads) {
    float* distances = (float*)malloc(sizeof(float) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = euclidean_distance_f32_avx(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("avx")))
double* centroid_f64_avx(const double** vecs, size_t n_vectors, size_t length) {
    double* centroid = (double*)malloc(sizeof(double) * length);

    for (size_t k = 0; k < length; k++) centroid[k] = 0.0;

    size_t i, j;

    for (j = 0; j + 4 <= length; j += 4) {
        __m256d sum = _mm256_setzero_pd();

        for (i = 0; i < n_vectors; i++) {
            __m256d v = _mm256_loadu_pd(&vecs[i][j]);
            sum = _mm256_add_pd(sum, v);
        }

        __m256d avg = _mm256_div_pd(sum, _mm256_set1_pd((double)n_vectors));
        _mm256_storeu_pd(&centroid[j], avg);
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

__attribute__((target("avx")))
float* centroid_f32_avx(const float** vecs, size_t n_vectors, size_t length) {
    float* centroid = (float*)malloc(sizeof(float) * length);

    for (size_t k = 0; k < length; k++) centroid[k] = 0.0f;

    size_t i, j;

    for (j = 0; j + 8 <= length; j += 8) {
        __m256 sum = _mm256_setzero_ps();

        for (i = 0; i < n_vectors; i++) {
            __m256 v = _mm256_loadu_ps(&vecs[i][j]);
            sum = _mm256_add_ps(sum, v);
        }

        __m256 avg = _mm256_div_ps(sum, _mm256_set1_ps((float)n_vectors));
        _mm256_storeu_ps(&centroid[j], avg);
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

__attribute__((target("avx")))
double manhattan_distance_f64_avx(const double* vec_a, const double* vec_b, const size_t length) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;

    // Mask to zero the sign bit (0x7FFFFFFFFFFFFFFF)
    const __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    for (; i + 4 <= length; i += 4) {
        __m256d va = _mm256_loadu_pd(vec_a + i);
        __m256d vb = _mm256_loadu_pd(vec_b + i);
        __m256d diff = _mm256_sub_pd(va, vb);
        __m256d abs_diff = _mm256_and_pd(diff, sign_mask);
        sum = _mm256_add_pd(sum, abs_diff);
    }

    double buffer[4];
    _mm256_storeu_pd(buffer, sum);
    double distance = buffer[0] + buffer[1] + buffer[2] + buffer[3];

    for (; i < length; i++) {
        double diff = vec_a[i] - vec_b[i];
        distance += fabs(diff);
    }

    return distance;
}

__attribute__((target("avx")))
float manhattan_distance_f32_avx(const float* vec_a, const float* vec_b, const size_t length) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    // Sign-bit mask for floats: 0x7FFFFFFF
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    for (; i + 8 <= length; i += 8) {
        __m256 va = _mm256_loadu_ps(vec_a + i);
        __m256 vb = _mm256_loadu_ps(vec_b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 abs_diff = _mm256_and_ps(diff, sign_mask); // clear sign bit
        sum = _mm256_add_ps(sum, abs_diff);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                     buffer[4] + buffer[5] + buffer[6] + buffer[7];

    // handle remainder
    for (; i < length; ++i) {
        distance += fabsf(vec_a[i] - vec_b[i]);
    }

    return distance;
}

__attribute__((target("avx")))
double* batched_manhattan_f64_avx(const double** vecs, const double* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    double* distances = (double*)malloc(sizeof(double) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = manhattan_distance_f64_avx(vecs[i], vec, length);
        }
    }

    return distances;
}

__attribute__((target("avx")))
float* batched_manhattan_f32_avx(const float** vecs, const float* vec, const size_t n_vectors, const size_t length, size_t num_threads) {
    float* distances = (float*)malloc(sizeof(float) * n_vectors);
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < n_vectors; i++) {
            distances[i] = manhattan_distance_f32_avx(vecs[i], vec, length);
        }
    }

    return distances;
}