extern void _bazel_runfiles_init_impl(int argc, char **argv, char **envp);

// Preserve the existing Linux startup ABI. Only the registration pointer moves
// to C; runfiles discovery and initialization remain in Zig.
void (*const _bazel_runfiles_init)(int, char **, char **)
    __attribute__((section(".init_array"), used)) = _bazel_runfiles_init_impl;
