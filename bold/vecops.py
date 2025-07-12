import os
import numpy as np
import ctypes


class Vecop:
    def __init__(self, libpath: str, precision: str = "float", arch: str = "scalar") -> None:
        if precision.lower() not in {"float", "double"}:
            raise ValueError(f"{precision} invalid. Try 'float' or 'double'.")
        self.precision = precision.lower()

        if not os.path.exists(libpath):
            raise FileNotFoundError(f"{libpath} does not exist.")

        self.lib = ctypes.CDLL(libpath)
        self.arch = arch

        # Set function signatures
        if self.precision == "float":
            self.c_type = ctypes.c_float
            suffix = "f32"
        else:
            self.c_type = ctypes.c_double
            suffix = "f64"

        self.vec_ptr_type = ctypes.POINTER(self.c_type)

        self.euclidean_distance_func = getattr(self.lib, f"euclidean_distance_{suffix}")
        self.batched_euclidean_func = getattr(self.lib, f"batched_euclidean_{suffix}")
        self.centroid_func = getattr(self.lib, f"centroid_{suffix}")
        self.manhattan_distance_func = getattr(self.lib, f"manhattan_distance_{suffix}")
        self.batched_manhattan_func = getattr(self.lib, f"batched_manhattan_{suffix}")

        self.get_arch = self.lib.vecop_print_arch
        self.get_arch.restype = ctypes.c_char_p

        self.set_arch = self.lib.vecop_set_arch
        self.set_arch.argtypes = [ctypes.c_char_p]

        # Scalar functions
        self.euclidean_distance_func.argtypes = [self.vec_ptr_type, self.vec_ptr_type, ctypes.c_size_t]
        self.euclidean_distance_func.restype = self.c_type

        self.manhattan_distance_func.argtypes = [self.vec_ptr_type, self.vec_ptr_type, ctypes.c_size_t]
        self.manhattan_distance_func.restype = self.c_type

        # Batched functions
        self.batched_euclidean_func.argtypes = [
            ctypes.POINTER(self.vec_ptr_type), self.vec_ptr_type,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
        ]
        self.batched_euclidean_func.restype = self.vec_ptr_type

        self.batched_manhattan_func.argtypes = [
            ctypes.POINTER(self.vec_ptr_type), self.vec_ptr_type,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
        ]
        self.batched_manhattan_func.restype = self.vec_ptr_type

        # Centroid
        self.centroid_func.argtypes = [ctypes.POINTER(self.vec_ptr_type), ctypes.c_size_t, ctypes.c_size_t]
        self.centroid_func.restype = self.vec_ptr_type

        self.change_architecture(self.arch)

    def _normalize_arch(self, arch: str) -> str:
        arch = arch.lower().replace("_", "-")
        # Accept avx512, avx-512, avx_512 as the same
        if arch in {"avx512", "avx-512", "avx_512"}:
            return "AVX512"
        if arch in {"avx2"}:
            return "AVX2"
        if arch in {"avx"}:
            return "AVX"
        if arch in {"sse"}:
            return "SSE"
        if arch in {"scalar"}:
            return "SCALAR"
        raise ValueError(f"{arch} is invalid. Use one of: 'sse', 'avx', 'avx2', 'avx512', 'scalar'")

    def change_architecture(self, arch: str = "sse") -> None:
        norm_arch = self._normalize_arch(arch)
        self.arch = norm_arch
        self.set_arch(norm_arch.encode())

    def architecture(self) -> str:
        result = self.get_arch()
        if isinstance(result, bytes):
            return result.decode("utf-8")
        return str(result)

    def get_pointer(self, vec: np.ndarray):
        return ctypes.cast(vec.ctypes.data, self.vec_ptr_type)

    def get_vector_pointer_array(self, vecs: np.ndarray):
        """Precompute array of vector pointers"""
        n_vecs = vecs.shape[0]
        return (self.vec_ptr_type * n_vecs)(
            *[self.get_pointer(vec) for vec in vecs]
        )

    def euclidean_distance(self, ptr_a, ptr_b, length: int) -> float:
        return self.euclidean_distance_func(ptr_a, ptr_b, length)

    def manhattan_distance(self, ptr_a, ptr_b, length: int) -> float:
        return self.manhattan_distance_func(ptr_a, ptr_b, length)

    def euclidean_distance_batched(self, vec_pointers, ptr_target, n_vecs: int, length: int, num_threads: int = 1) -> np.ndarray:
        res_ptr = self.batched_euclidean_func(vec_pointers, ptr_target, n_vecs, length, num_threads)
        return np.ctypeslib.as_array(res_ptr, shape=(n_vecs,))

    def manhattan_distance_batched(self, vec_pointers, ptr_target, n_vecs: int, length: int, num_threads: int = 1) -> np.ndarray:
        res_ptr = self.batched_manhattan_func(vec_pointers, ptr_target, n_vecs, length, num_threads)
        return np.ctypeslib.as_array(res_ptr, shape=(n_vecs,))

    def centroid(self, vec_pointers, n_vecs: int, length: int) -> np.ndarray:
        res_ptr = self.centroid_func(vec_pointers, n_vecs, length)
        return np.ctypeslib.as_array(res_ptr, shape=(length,))
