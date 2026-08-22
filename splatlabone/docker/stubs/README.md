# libcuda.so.1 — Kaniko build stub

Kaniko has no NVIDIA driver. CUDA base images need `libcuda.so.1` before any `RUN`
(because `/bin/sh` fails with exit 127 otherwise).

`COPY --from=<cuda-devel>` works but forces Kaniko to unpack the multi-GB devel
image again (often 30–60+ minutes). This tiny ELF in the build context unblocks
`/bin/sh` instantly; the first `RUN` then `ln -sf`s the real toolkit stub from
`/usr/local/cuda/lib64/stubs/libcuda.so` already present in the base image.
