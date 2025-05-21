def make_jit_source(name, deps = [], **kwargs):
    """
    Bazel equivalent of the CMake make_jit_source function.
    Generates a C++ source file from a Metal header file by running a script.
    
    Args:
        name: The Metal header file to process.
        deps: Additional dependencies required for processing.
        kwargs: Additional arguments to pass to the native.genrule function.
    """
    src_name = name.split("/")[-1]  # Extract the filename
    output_file = "mlx/backend/metal/jit/{}.cpp".format(src_name)

    native.genrule(
        name = name,
        srcs = ["mlx/backend/metal/kernels/{}.h".format(src_name)] + deps,
        outs = [output_file],
        cmd = (
            "bash $(location :mlx/backend/metal/make_compiled_preamble.sh) $@ $(location mlx/backend/metal/kernels/{}.h)"
        ).format(name),
        tools = [":mlx/backend/metal/make_compiled_preamble.sh"],
        **kwargs,
    )
