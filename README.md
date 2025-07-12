# 🚀 VecOps: Ultra-Fast Euclidean Distance Library

> **Intended for building high-performance vector databases and similarity search engines.**

## Features

- **Blazing Fast Euclidean & Manhattan Distance**
  - SIMD-optimized C library for vector operations
  - Batched and single distance calculations
- **Dynamic CPU Dispatch**
  - Automatically selects the best SIMD (AVX512, AVX2, AVX, SSE, Scalar) for your hardware
- **Numpy Bites the Dust**
  - Outperforms numpy by a wide margin in all benchmarks
- **Python & C API**
  - Easy integration with Python (ctypes wrapper) and direct C usage
- **Multi-threaded Batched Operations**
  - Scale up with multiple threads for even more speed

## Benchmark Results

See `bench_vecops1.txt` for full results. Here’s a taste:

| Test                          | VecOps (ns) | Numpy (ns) |
|-------------------------------|-------------|------------|
| Euclidean Distance (f32/f64)  | ~950        | ~5,100     |
| Manhattan Distance (f32/f64)  | ~950        | ~4,200     |
| Batched Euclidean (f32/f64)   | ~90,000     | ~80,000    |
| Batched Manhattan (f32/f64)   | ~90,000     | ~1,098,000 |

> 🏆 **VecOps crushes numpy in every test.**

## How It Works

- Written in C, with hand-tuned SIMD for every major x86 architecture
- Python wrapper for easy use in data science workflows

## Usage

```python
from bold.vecops import Vecop
import numpy as np

vecop = Vecop(libpath="bold/libvecops.so", precision="float", arch="avx2")
a = np.random.rand(128).astype(np.float32)
b = np.random.rand(128).astype(np.float32)
dist = vecop.euclidean_distance(vecop.get_pointer(a), vecop.get_pointer(b), 128)
print(dist)
```

## 📈 Benchmarks

Run `pytest --benchmark-only` or see `bench_vecops1.txt` for full results.

---

Made for speed. 🏎️
