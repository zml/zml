extern void _bazel_runfiles_init_impl(int argc, char **argv, char **envp);

// Preserve the existing Linux startup ABI. Registration happens in C;
// runfiles discovery and initialization remain in Zig.
// Requires glibc's argc/argv/envp constructor arguments; musl does not pass them.
// The original Zig initializer already made this assumption to read argv[0].
__attribute__((constructor))
void _bazel_runfiles_init(int argc, char **argv, char **envp) {
    _bazel_runfiles_init_impl(argc, argv, envp);
}
