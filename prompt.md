You are an expert OpenAI Triton, AWS Neuron Kernel Interface (NKI), and Zig developer working within the ZML framework. Your task is to translate a Triton-based neural network implementation into an optimized AWS Neuron NKI kernel, and write the corresponding Zig host code to execute and benchmark it.

**Context & File Paths:**
* Reference Triton Implementation Directory: `/home/kevin/monorepo/llmd/models/triton`
* Reference Kernel Generator: `/home/kevin/monorepo/llmd/models/triton/llmd_triton_generate.py`
* Reference Triton Test Kernel: `/home/kevin/monorepo/llmd/models/triton/triton_kernels/test_kernel.py`
* Reference NKI/Zig Examples (for style/patterns): `/home/kevin/zml/examples/neuron_nki/add.py` and `/home/kevin/zml/examples/neuron_nki/add.zig`
* Target Python Kernel Output: `/home/kevin/zml/examples/neuron_nki/gated_deltanet.py`
* Target Zig Host Output: `/home/kevin/zml/examples/neuron_nki/gated_deltanet.zig`

Please fulfill the following requirements:

**Step 1: NKI Kernel Implementation (`gated_deltanet.py`)**
1. Read the reference Triton kernel logic from the provided `triton` directory.
2. Translate the Triton block mechanics (e.g., pointers, block memory accesses, math) into equivalent AWS Neuron NKI semantics using `neuronx.nki.language` (e.g., `nl.load`, `nl.store`, tile allocations).
3. Structure your Python file to perfectly match the patterns and decorators used in the reference `add.py`.

**Step 2: Zig Host & Validation Code (`gated_deltanet.zig`)**
1. Create the Zig host code using the ZML framework, mirroring the structural patterns in `add.zig`.
2. Ensure the Zig code loads the `gated_deltanet.py` NKI kernel correctly.
3. Write a test block inside the Zig file that:
   - Initializes input tensors with appropriate shapes and standard dtypes (e.g., `bfloat16` or `float32`).
   - Computes a basic mathematical baseline in pure Zig/ZML to serve as the ground truth.
   - Executes the NKI kernel.
   - Compares the outputs of the NKI kernel against the ground truth using a reasonable floating-point tolerance suited for Neuron hardware.
   - Benchmarks the execution speed.

**Step 3: Build & Execution Configuration**
Assume the project uses Bazel. Ensure your code requires no external dependencies outside of standard ZML and Neuron tools. 

The implementation must be fully valid so that running the following command compiles, tests, and benchmarks the kernel without errors:
> NEURON_RT_LOG_LEVEL=debug NEURON_RT_VISIBLE_CORES=1 bazel run --config=remote --@zml//platforms:neuron=true --@zml//platforms:cpu=false //examples/neuron_nki:gated_deltanet

----


 alway use remote build ex:  bazel run --config=remote


----


based on `/home/kevin/zml/zml/nn.zig` implement the same `gated delta net` test in /home/kevin/zml/examples/neuron_nki/

----
