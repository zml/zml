# Troubleshooting

The Bazel build system should install all necessary dependencies including `llvm` and `zig`. Issues can arise from conflicts with local toolchains.

## MacOS

Useful Bazel commands:

- `bazel fetch --force --configure`: this will refetch and configure local sandbox for external toolchains.
- `bazel clean --expunge`: this will remove all downloaded dependencies and start fresh.

Make sure that your MacOS is up to date. There is a known issue with `14.7`.

Verify the existance of the Xcode CLI developer tools: `xcode-select --install`.
Reinstall the CLT:

```bash
sudo rm -rf /Library/Developer/CommandLineTools && \
sudo xcode-select -r && \
sudo xcode-select --install
```
