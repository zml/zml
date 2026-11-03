extern void _bazel_runfiles_init(int argc, char **argv, char **envp);

// Zig's incremental ELF linker supports constructor sections from C objects,
// but currently drops a Zig declaration's .init_array section on a type conflict.
__attribute__((constructor)) static void init_runfiles(int argc, char **argv, char **envp) {
    _bazel_runfiles_init(argc, argv, envp);
}
