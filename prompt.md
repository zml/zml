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

You are an expert AWS Neuron NKI and Zig developer working within the ZML framework. Your task is to update an existing GatedDeltaNet implementation to match the Qwen3.5-9B model architecture and optimize the NKI kernel using official AWS reference patterns.

**Context & Resources:**
* Target Zig Host: `/home/kevin/zml/examples/neuron_nki/gated_deltanet.zig`
* Target Python Kernel: `/home/kevin/zml/examples/neuron_nki/gated_deltanet.py`
* Architecture Config: [https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json](https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json)
* NKI Optimization Reference: [https://raw.githubusercontent.com/aws-neuron/nki-samples/refs/heads/main/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py](https://raw.githubusercontent.com/aws-neuron/nki-samples/refs/heads/main/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py)
* NKI Fallback Docs: [https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/_sources/nki/programming_model.rst](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/_sources/nki/programming_model.rst)

Please execute the following steps:

**Step 1: Update the Zig Configuration**
Fetch the Qwen3.5-9B `config.json`. Based on its parameters (like `num_attention_heads`, `num_key_value_heads`, and `hidden_size`), calculate and update the following constants in `/home/kevin/zml/examples/neuron_nki/gated_deltanet.zig`:
```zig
const batch_size = 1;
const seq_len = 16;
const num_q_heads = [extract from config];
const num_value_heads = [extract from config];
const key_dim = [calculate from config];
const value_dim = [calculate from config];
```

**Step 2: Optimize the NKI Kernel**
Review the provided `mamba_nki_kernels.py` reference file. Refactor the NKI kernel in `gated_deltanet.py` to strictly mirror its coding style and optimization techniques. 
* Pay special attention to tiling strategies, loop unrolling, and optimal `nl.load`/`nl.store` usage for Neuron hardware.
* If you encounter any API constraints or errors while optimizing, consult the provided Neuron NKI programming model documentation link to resolve them.

**Step 3: Validation Command**
Ensure the final output is completely self-contained, requires no external dependencies outside of standard ZML/Neuron libraries, and executes flawlessly using the following command:
> NEURON_RT_LOG_LEVEL=error NEURON_RT_VISIBLE_CORES=1 bazel run --config=remote --@zml//platforms:neuron=true --@zml//platforms:cpu=false //examples/neuron_nki:gated_deltanet

----

You are an expert AWS Neuron NKI kernel developer optimizing code for Inferentia2 (inf2.8xlarge). 

**Current Situation:**
The current GatedDeltaNet NKI kernel implementation located at `examples/neuron_nki/gated_deltanet.py` suffers from extremely slow compile times and poor runtime performance. 

**Your Task:**
Rework the `gated_deltanet.py` NKI implementation from scratch.

**Reference Material:**
Study the state-of-the-art NKI optimization techniques and memory management strategies used in this official reference:
[https://raw.githubusercontent.com/aws-neuron/nki-samples/refs/heads/main/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py](https://raw.githubusercontent.com/aws-neuron/nki-samples/refs/heads/main/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py)

**Constraints & Architecture:**
1. **Hardware:** `inf2.8xlarge` targeting exactly ONE visible Neuron core (`NEURON_RT_VISIBLE_CORES=1`).
2. **Neuron Architecture:** Deeply respect the Neuron memory hierarchy (HBM, SBUF, PSUM). Optimize your `nl.load` and `nl.store` patterns, DMA transfers, and use appropriate spatial/computational tiling to maximize core utilization without exhausting SBUF limits.
3. **Compile Time:** Avoid excessive static loop unrolling or massive inline blocks that cause the Neuron compiler to hang or bloat the LLVM IR. Use `nl.affine_range` and structured loops to keep compile times fast.
4. **Runtime:** The execution of your kernel must be faster than the baseline `ReferenceProgram`.

**Context & Resources:**
* Target Zig Host: `/home/kevin/zml/examples/neuron_nki/gated_deltanet.zig`
* Target Python Kernel: `/home/kevin/zml/examples/neuron_nki/gated_deltanet.py`
* Architecture Config: [https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json](https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json)
* NKI Optimization Reference: [https://raw.githubusercontent.com/aws-neuron/nki-samples/refs/heads/main/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py](https://raw.githubusercontent.com/aws-neuron/nki-samples/refs/heads/main/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py)
* NKI Fallback Docs: [https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/_sources/nki/programming_model.rst](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/_sources/nki/programming_model.rst)

**Validation Command:**
Your code must execute cleanly via:
`NEURON_RT_LOG_LEVEL=info NEURON_RT_VISIBLE_CORES=1 bazel run --config=remote --@zml//platforms:neuron=true --@zml//platforms:cpu=false //examples/neuron_nki:gated_deltanet`

**Deliverables:**
1. **Primary Output:** The completely rewritten `examples/neuron_nki/gated_deltanet.py` file containing the optimized kernel.
2. **Fallback Output (`followup.md`):** If achieving a runtime faster than the `ReferenceProgram` is theoretically or practically impossible due to GatedDeltaNet's specific memory access patterns, Neuron hardware constraints, or NKI API limitations, you must generate a `followup.md` file instead of (or alongside) the kernel. This document must explain the exact architectural bottlenecks (e.g., memory bandwidth, sequential dependency limits, PSUM constraints) preventing the speedup, backed by explanations of the Neuron programming model and relevant source documentation.
 
**Validation Command**
Ensure the final output is completely self-contained, requires no external dependencies outside of standard ZML/Neuron libraries, and executes flawlessly using the following command:
> NEURON_RT_LOG_LEVEL=error NEURON_RT_VISIBLE_CORES=1 bazel run --config=remote --@zml//platforms:neuron=true --@zml//platforms:cpu=false //examples/neuron_nki:gated_deltanet


---



You can use 

bazel run --config=remote --@zml//platforms:neuron=true --run_under="NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=/home/kevin/profiling NEURON_RT_INSPECT_SYSTEM_PROFILE=1 NEURON_FRAMEWORK_DEBUG=1 NEURON_RT_INSPECT_DEVICE_PROFILE=1 XLA_HLO_DEBUG=1 XLA_IR_DEBUG=1 neuron-explorer inspect -o /home/kevin/profiling/ -- " --@zml//platforms:cpu=false //examples/neuron_nki:gated_deltanet

to run it with profiling use

neuron-explorer view -d /home/kevin/profiling/... --output-formatsummary-text


---

You are an expert AWS Neuron NKI (Neuron Kernel Interface) developer and Machine Learning Systems Engineer specializing in hardware optimization for AWS Inferentia2 (inf2.8xlarge) and Trainium. 

I am investigating the performance trade-offs, dispatch overhead, and scaling limits of writing custom NKI kernels exposed as JAX XLA custom calls, compared to relying on pure native JAX implementations on the Neuron platform.

Below, I will provide:
1. A sample script (`run_matmul.py`) containing both a basic single-tile NKI matmul and a pure JAX matmul.
2. The profiling summary output generated from `neuron-profile view --output-format summary-text`.

Based on this data and your expert knowledge of the Neuron architecture, please provide a comprehensive technical report addressing the following objectives:

### 1. Dispatch Overhead & Base Cost Analysis
Analyze the profiling summary to quantify the overhead of dispatching the NKI kernel via a JAX XLA custom call versus the native JAX execution. Explain where the time is being spent (e.g., framework overhead, DMA transfers, instruction decoding, vs. actual Matrix Engine compute).

### 2. The "Single-Tile Limit" and Scaling (M, K, N > 64, 128, 512)
The basic NKI kernel states: "We must stay within the single-tile limits for this basic kernel." 
* Explain what happens at the hardware level if we exceed these dimensions (e.g., SBUF/PSUM capacity limits).
* Compare the scaling paths: If we scale to larger matrices, does native JAX inherently become faster because it handles tiling/blocking automatically, or does NKI remain competitive? 
* Assess the "development cost vs. compute cost" of having to write a fully blocked, tiled, and loop-optimized NKI matmul for larger shapes.

### 3. Mandatory Optimizations
To achieve maximum TFLOPS on the Neuron Matrix Multiplication Engine for large matrices, what explicit optimizations are *mandatory* when writing NKI code? (Please touch on concepts like SRAM/SBUF double-buffering, DMA pipelining, and layout/transposition constraints).

### 4. Final Verdict: When is NKI worth it?
Provide a clear heuristic or rule-of-thumb on when an engineering team should invest time writing custom NKI kernels versus just letting the JAX XLA compiler handle the workload on Neuron. Under what specific conditions does NKI provide a clear ROI?

---

the code is in /home/kevin/zml/run_matmul.py

source .venv/bin/activate

NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=/home/kevin/profiling NEURON_RT_INSPECT_SYSTEM_PROFILE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_FRAMEWORK_DEBUG=1 XLA_HLO_DEBUG=1 XLA_IR_DEBUG=1 python run_matmul.py
