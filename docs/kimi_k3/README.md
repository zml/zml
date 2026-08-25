# Kimi K3 port controls

This directory holds versioned provenance for the NVIDIA-only Kimi K3 port.
The executable plan and milestone SITREPs live at the active workspace root,
while implementation work and focused commits are made in this ZML repository.

Rules enforced by the project plan:

- full-model compilation, staging, and historical conformance remain pinned to
  physical GPU 0; Milestones 22–26 additionally allow physical GPUs 0–3 for
  their isolated four-layer diagnostic and distributed tests;
- the normal Kimi `//examples/llm` command runs a 47-layer TP4+EP4 resident
  diagnostic on four CUDA devices or all 93 layers with TP8+EP8 on eight;
  TP and EP share the same physical ranks, and generated facts remain marked
  unreliable in both modes;
- CPU inference fallback is forbidden;
- model weights are never downloaded by project scripts;
- checkpoint input is read from the workspace-local
  `moonshot/kimi-k3` directory;
- all 96 user-supplied shards are present and Milestone 21 verifies them offline;
- every important change is committed separately with its validation evidence.

`revisions.lock.json` is self-contained so the external `upstream/` directory
can be deleted. The ZML donor commits remain available as remote refs in this
repository, and Moonshot source/checkpoint identity is enforced by SHA-256.

Implementation evidence:

- [Native MXFP4 comparison](native-mxfp4-comparison.md)
- [Optimized KDA/MLA comparison](optimized-kda-mla-comparison.md)
- [Full-model and distributed readiness](full-model-readiness.md)
- [Permanent conformance and operations](permanent-conformance.md)
- [Four-layer GPU0 and TP4 operations](four-gpu-prefix-operations.md)
