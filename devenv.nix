{
  pkgs,
  config,
  ...
}: {
  packages = with pkgs; [
    bazelisk
    python3
    uv
    zig
  ];
  env.ZLS_CMD = "${config.devenv.root}/tools/zls.sh";
  scripts.bazel.exec = ''bazelisk "$@"'';

  enterShell = pkgs.lib.optionalString pkgs.stdenv.isDarwin ''
    # Use the "real" XCode rather than Nixpkg's one. This does assume you have it installed.
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    export SDKROOT="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk";
  '';
}
