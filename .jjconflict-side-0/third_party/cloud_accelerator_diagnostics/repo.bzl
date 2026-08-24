_COMMIT = "6eccaf47364904d1c55209dd8795e3bf9cac21e1"
_URL_PREFIX = "https://raw.githubusercontent.com/AI-Hypercomputer/cloud-accelerator-diagnostics/" + _COMMIT + "/tpu_info/tpu_info/proto/"

_PROTOS = {
    "tpu_metric_service.proto": "a89d80938c75177ff45fb4c6130f5d71bc5164df473a77948e31b7dedd96a1bb",
    "tpu_telemetry.proto": "4ad6096e40de5ba00b6c7badea1218c801069b82be829ec56dc0ce693b5a2505",
}

def _impl(ctx):
    for output, sha256 in _PROTOS.items():
        ctx.download(
            url = _URL_PREFIX + output,
            sha256 = sha256,
            output = output,
        )
    ctx.template("BUILD.bazel", ctx.attr.build_file)

_cloud_accelerator_diagnostics = repository_rule(
    implementation = _impl,
    attrs = {
        "build_file": attr.label(mandatory = True),
    },
)

def repo():
    _cloud_accelerator_diagnostics(
        name = "cloud_accelerator_diagnostics",
        build_file = "//third_party/cloud_accelerator_diagnostics:cloud_accelerator_diagnostics.BUILD.bazel",
    )
