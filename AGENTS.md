# Overview

You're working on a CLI client to display prometheus metrics.

* Use bazel to build the client: `bazel build //bin/zml-smi/prometheus`
* Client Code is under `bin/zml-smi/prometheus`
* CLI widgets are defined under `bin/zml-smi/tui`
* Use curl to inspect actual metrics from `http://gh200:8001/metrics`
* Bazel uses Zig 0.16. 
* find the current Zig standard library used by Bazel: `echo "$(bazel info output_base)/$(bazel cquery --output=files "$(bazel cquery "filter('zig_toolchain', deps(//zml))" 2>/dev/null | cut -d' ' -f1 | head -n1)" 2>/dev/null | rg "/lib\$")/std"`
