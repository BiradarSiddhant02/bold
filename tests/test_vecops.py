import os
import sys
# Add project root to sys.path for bold import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pytest
from bold.vecops import Vecop

@pytest.fixture(scope="module")
def vecop(request):
    # Get architecture from command line option
    arch = request.config.getoption("--arch")
    # Use double precision for more robust tests
    libpath = os.path.join(os.path.dirname(__file__), "..", "bold", "libvecops.so")
    libpath = os.path.abspath(libpath)
    v = Vecop(libpath=libpath, precision="double", arch=arch)
    return v

def test_architecture(vecop):
    # Should match the requested architecture (case-insensitive, allow dash/underscore)
    arch = vecop.architecture()
    if isinstance(arch, bytes):
        arch = arch.decode()
    arch = arch.lower().replace("-", "")
    expected = vecop.arch.lower().replace("-", "")
    assert arch == expected or expected in arch

def test_euclidean_distance(vecop):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 6.0, 8.0], dtype=np.float64)
    ptr_a = vecop.get_pointer(a)
    ptr_b = vecop.get_pointer(b)
    dist = vecop.euclidean_distance(ptr_a, ptr_b, 3)
    expected = np.linalg.norm(a - b)
    assert np.isclose(dist, expected)

def test_euclidean_distance_batched(vecop):
    vecs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    ref = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vec_ptrs = vecop.get_vector_pointer_array(vecs)
    ptr_ref = vecop.get_pointer(ref)
    dists = vecop.euclidean_distance_batched(vec_ptrs, ptr_ref, 2, 3)
    expected = np.linalg.norm(vecs - ref, axis=1)
    assert np.allclose(dists, expected)

def test_centroid(vecop):
    vecs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    vec_ptrs = vecop.get_vector_pointer_array(vecs)
    centroid = vecop.centroid(vec_ptrs, 2, 3)
    expected = np.mean(vecs, axis=0)
    assert np.allclose(centroid, expected)

def test_manhattan_distance(vecop):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 6.0, 8.0], dtype=np.float64)
    ptr_a = vecop.get_pointer(a)
    ptr_b = vecop.get_pointer(b)
    dist = vecop.manhattan_distance(ptr_a, ptr_b, 3)
    expected = np.sum(np.abs(a - b))
    assert np.isclose(dist, expected)

def test_manhattan_distance_batched(vecop):
    vecs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    ref = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vec_ptrs = vecop.get_vector_pointer_array(vecs)
    ptr_ref = vecop.get_pointer(ref)
    dists = vecop.manhattan_distance_batched(vec_ptrs, ptr_ref, 2, 3)
    expected = np.sum(np.abs(vecs - ref), axis=1)
    assert np.allclose(dists, expected)

def test_dynamic_architecture_change(vecop):
    # Define SIMD arch priority
    arch_priority = {"scalar": 0, "sse": 1, "avx": 2, "avx2": 3, "avx512": 4}
    for arch in ["scalar", "sse", "avx", "avx2"]:
        vecop.change_architecture(arch)
        current = vecop.architecture()
        if isinstance(current, bytes):
            current = current.decode()
        current = current.lower().replace("-", "")
        requested = arch.lower().replace("-", "")
        # Accept if current arch is at least as high as requested
        assert arch_priority.get(current, 0) >= arch_priority.get(requested, 0), f"Requested {requested}, got {current}"
    # Restore original
    vecop.change_architecture(vecop.arch)

def pytest_addoption(parser):
    parser.addoption(
        "--arch",
        action="store",
        default="scalar",
        help="SIMD architecture to use: sse, avx, avx2, avx-512, scalar",
    )
