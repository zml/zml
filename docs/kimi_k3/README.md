# Kimi K3 port controls

This directory holds versioned provenance for the NVIDIA-only Kimi K3 port.
The executable plan and milestone SITREPs remain at `/ephemeral/kimi-k3` as
requested, while implementation work and focused commits are made in this ZML
repository.

Rules enforced by the project plan:

- inference and performance validation use NVIDIA GPUs only;
- CPU inference fallback is forbidden;
- model weights are never downloaded by project scripts;
- checkpoint input is read from `/ephemeral/kimi-k3/moonshot/kimi-k3`;
- missing full-checkpoint shards are supplied by the user later;
- every important change is committed separately with its validation evidence.

`revisions.lock.json` is self-contained so the external `upstream/` directory
can be deleted. The ZML donor commits remain available as remote refs in this
repository, and Moonshot source/checkpoint identity is enforced by SHA-256.
