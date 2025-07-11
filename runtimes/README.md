# Runtimes

This package contains the various runtimes ZML supports.
Each subpackage contains the logic to setup and load PJRT plugins for a given runtime.

## PJRT Plugin Sandboxing

PJRT plugins use `dlopen` to load shared libraries at runtime.
One of our goal is to sandbox the entire dynamic dependency set so that it is fully hermetic.

For this, our approach is the following:
1.	Flatten the dependency tree.

We collect all transitive shared library dependencies of the PJRT plugin into a single sandbox/ directory.

2.	Set relative `RPATH`.

We patch all ELF binaries and shared objects in the sandbox so that their `RPATH` = `$ORIGIN`. This ensures all dependencies are resolved relative to their location in the sandbox.

3.	Ensure `dlopen` visibility.

Since `dlopen` only consults `RPATH` if the target library was listed as a `NEEDED` dependency at link time, we also explicitly add any dynamically-loaded libraries to the plugin’s `NEEDED` section.
