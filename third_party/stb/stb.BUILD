cc_import(
    name = "stb_image_write",
    hdrs = ["stb_image_write.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

genrule(
    name = "impl",
    outs = ["impl.c"],
    cmd = """\
cat <<EOF > $(@)
#include "stb_image.h"
#include "stb_image_resize2.h"
#include "stb_image_write.h"
EOF
""",
)

cc_library(
    name = "stb",
    local_defines = [
        "STB_IMAGE_IMPLEMENTATION",
        "STB_IMAGE_RESIZE_IMPLEMENTATION",
        "STB_IMAGE_WRITE_IMPLEMENTATION",
    ],
    srcs = [":impl"],
    hdrs = [
        "stb_image.h",
        "stb_image_resize2.h",
        "stb_image_write.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
