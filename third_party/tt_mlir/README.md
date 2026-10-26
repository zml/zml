# TT-MLIR frontend dialects

Pinned to the same TT-MLIR revision as the Tenstorrent XLA plugin. This target
builds TTIR and TTCore with ZML's LLVM/MLIR; device execution and optimization
remain in the plugin. It does not link Metalium, TTNN, or another MLIR runtime.

`llvm.patch` carries the IR portions of the plugin's LLVM compatibility port.
`standalone_ttir.patch` replaces broad dialect includes with direct dependencies
and checks TTNN layout attributes by their registered names. The latter avoids
linking TTNN's attribute TypeIDs into the standalone TTIR frontend. These names
come from the same generated declarations, and null encodings remain rejected.

FlatBuffers is fetched for its native C++ targets only. Its Bzlmod registration
also brings unrelated JavaScript toolchains incompatible with this workspace's
Bazel version.
