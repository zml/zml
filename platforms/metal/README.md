# Bundled Metal Toolchain

ZML bundles Apple's Metal compiler tools (`metal`, `air-opt`, `air-as`, `metallib`) and Metal standard library headers via `http_aea_archive(name = "metal_toolchain", ...)` in `//MODULE.bazel`.

## Updating `@metal_toolchain`

### 1. Find the target build ID

Fetch Apple's Xcode downloadable index plist at:
`https://devimages-cdn.apple.com/downloads/xcode/simulators/index2.dvtdownloadableindex`

Look in the `otherDownloadables` array for entries with `assetType` equal to `metalToolchain`. They are ordered oldest to newest:
- GA releases have a short `assetBuildUpdate` (such as `27B47`) and no `_bN` beta suffix in `identifier` (betas look like `27C5035i` / `..._27_3_b2`).
- Record the chosen `assetBuildUpdate` value.

### 2. Fetch the URL, decryption key, and SHA-256

Query Apple's Pallas MobileAsset endpoint with a `POST` to `https://gdmf.apple.com/v2/assets` (`Content-Type: application/json`, skipping TLS CA verification if your client does not trust Apple's internal CA) with a JSON body containing:
- `AssetType`: `"com.apple.MobileAsset.MetalToolchain"`
- `AssetAudience`: `"02d8e57e-dd1c-4090-aa50-b4ed2aef0062"`
- `ClientVersion`: `2`
- `RequestedBuild`: the `assetBuildUpdate` string from step 1
- Standard Mac client fields: `CertIssuanceDay` (today's `YYYY-MM-DD`), `ProductVersion`, `BuildVersion`, `DeviceName` (`"Mac"`), `HWModelStr` (e.g. `"J414cAP"`), `ProductType` (e.g. `"Mac14,5"`)

The response is a JWT (`header.payload.signature`). Base64url-decode the middle `payload` segment to get JSON with `Assets[0]`, and read:
- `Build` → `canonical_id` (`com.apple.MobileAsset.MetalToolchain-<Build>`)
- `ArchiveDecryptionKey` → `archive_decryption_key`
- `__BaseURL` + `__RelativePath` → `urls`
- `_Measurement-SHA256` → base64-decode to lowercase hex for `sha256` (also returned in hex as `x-amz-meta-digest-sha256` by an HTTP `HEAD` on the `.aar` URL)

**macOS shortcut:** if the target Xcode is already installed locally, run `xcodebuild -downloadComponent MetalToolchain` and read the same keys from `/System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain/com_apple_MobileAsset_MetalToolchain.xml`.

### 3. Update Bazel configuration and verify

1. Update `archive_decryption_key`, `canonical_id`, `sha256`, and `urls` on `http_aea_archive(name = "metal_toolchain", ...)` in `//MODULE.bazel`.
2. Build `//platforms/metal:metal_toolchain`.
   - If Apple changed the `usr/metal/<major>` or `lib/clang/<ver>` directory names, or `metallib`'s symlink target (`air-lld`), update `includes` in `//MODULE.bazel`, `//platforms/metal:metal_toolchain.BUILD.bazel`, and `replace_prefixes` in `//platforms/metal/BUILD.bazel`.
3. Verify the extracted binaries' minimum OS and system-only dylib dependencies (`vtool -show-build` and `otool -L`), and smoke-test the packaged `metal` (both `-std=metal3.2` and `-std=metal4.0` including `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`), `air-as`, `air-opt`, and `metallib`.
