# === Stage 1: Build native .so with full toolchain ===
FROM debian:bookworm-slim AS builder

# Install build dependencies (no cleanup needed as this is a builder stage)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    make

WORKDIR /build

# Copy only the files needed for building the .so to leverage caching
COPY Makefile exports.map ./
COPY include/ ./include/
COPY src/ ./src/

# Build the shared library and strip it
RUN make all && \
    strip bold/libvecops.so

# === Stage 2: Minimal runtime image ===
FROM python:3.12-slim

# Install runtime dependencies (single layer, cleaned immediately)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --no-compile numpy pytest pytest-benchmark

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /build/bold/libvecops.so ./bold/
COPY bold/vecops.py ./bold/
COPY tests/ ./tests/
COPY run_all.bash .

# Set permissions in one layer
RUN mkdir -p bold && \
    chmod +x run_all.bash

# Run all tests and benchmarks
CMD ["./run_all.bash"]