# Optional ARM64 GIN host transport

The CUDA runtime includes four unmodified Ubuntu ARM64 libraries for the
optional GIN host-memory transport. The archives and SHA-256 hashes are pinned
in `../rdma.packages.lock.json`: libnl3 version 3.7.0-0.3build1.1 and rdma-core
version 50.0-2ubuntu0.2. No host package installation is required.

Only libnl-3.so.200, libnl-route-3.so.200, libibverbs.so.1 and libmlx5.so.1
are included. Their SONAMEs and bytes match the individually validated
transport closure. Load them in that order using absolute paths with global
symbol visibility. Loading libmlx5 registers its provider directly; this path
does not need a host provider configuration or IBV_DRIVERS. It requires a
compatible system glibc (the pinned packages require glibc 2.38 or later).

Each package's unmodified copyright file is under its own directory in
`share/licenses/rdma`. GPL-2 and LGPL-2.1 contain the complete standard license
texts referenced by those copyright files; the latter governs libnl3 library
code. The libraries' source locations are recorded by their package notices:
https://github.com/thom311/libnl and https://github.com/linux-rdma/rdma-core.

This packaging does not preload the transport, initialize CUDA/NCCL, change
provider environment variables, or alter the CUDA plugin/device-module bytes.
