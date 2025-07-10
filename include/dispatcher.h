#ifndef VECOP_DISPATCH_H
#define VECOP_DISPATCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Export symbol macro for shared libraries
#ifndef VECOP_API
#define VECOP_API __attribute__((visibility("default")))
#endif

// === Euclidean Distance ===
VECOP_API double euclidean_distance_f64(const double* a, const double* b, size_t len);
VECOP_API float  euclidean_distance_f32(const float* a, const float* b, size_t len);

// === Batched Euclidean Distance ===
VECOP_API double* batched_euclidean_f64(const double** vecs, const double* ref, size_t n, size_t len, size_t num_threads);
VECOP_API float*  batched_euclidean_f32(const float** vecs, const float* ref, size_t n, size_t len, size_t num_threads);

// === Centroid Computation ===
VECOP_API double* centroid_f64(const double** vecs, size_t n, size_t len);
VECOP_API float*  centroid_f32(const float** vecs, size_t n, size_t len);

// === Manhattan Distance ===
VECOP_API double manhattan_distance_f64(const double* a, const double* b, size_t len);
VECOP_API float  manhattan_distance_f32(const float* a, const float* b, size_t len);

// === Batched Manhattan Distance ===
VECOP_API double* batched_manhattan_f64(const double** vecs, const double* ref, size_t n, size_t len, size_t num_threads);
VECOP_API float*  batched_manhattan_f32(const float** vecs, const float* ref, size_t n, size_t len, size_t num_threads);

// === Architecture ===
VECOP_API const char* vecop_print_arch(void);
VECOP_API void vecop_set_arch(const char* arch);

#ifdef __cplusplus
}
#endif

#endif // VECOP_DISPATCH_H
