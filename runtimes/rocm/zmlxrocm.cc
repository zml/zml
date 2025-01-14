#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <string>

#include "tools/cpp/runfiles/runfiles.h"

static void setup_runfiles(int argc, char **argv) __attribute__((constructor))
{
    using bazel::tools::cpp::runfiles::Runfiles;
    auto runfiles = std::unique_ptr<Runfiles>(Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY));

    auto HIPBLASLT_EXT_OP_LIBRARY_PATH =
        runfiles->Rlocation("hipblaslt-dev/lib/hipblaslt/library/hipblasltExtOpLibrary.dat");
    if (HIPBLASLT_EXT_OP_LIBRARY_PATH != "")
    {
        setenv("HIPBLASLT_EXT_OP_LIBRARY_PATH", HIPBLASLT_EXT_OP_LIBRARY_PATH.c_str(), 1);
    }

    auto HIPBLASLT_TENSILE_LIBPATH = runfiles->Rlocation("hipblaslt-dev/lib/hipblaslt/library");
    if (HIPBLASLT_TENSILE_LIBPATH != "")
    {
        setenv("HIPBLASLT_TENSILE_LIBPATH", HIPBLASLT_TENSILE_LIBPATH.c_str(), 1);
    }

    auto ROCBLAS_TENSILE_LIBPATH = runfiles->Rlocation("rocblas/lib/rocblas/library");
    setenv("ROCBLAS_TENSILE_LIBPATH", ROCBLAS_TENSILE_LIBPATH.c_str(), 1);

    auto ROCM_PATH = runfiles->Rlocation("libpjrt_rocm/sandbox");
    setenv("ROCM_PATH", ROCM_PATH.c_str(), 1);
}

extern "C" void *zmlxrocm_dlopen(const char *filename, int flags) __attribute__((visibility("default")))
{
    if (filename != NULL)
    {
        char *replacements[] = {
            "librocm-core.so",
            "librocm-core.so.1",
            "librocm_smi64.so",
            "librocm_smi64.so.7",
            "libhsa-runtime64.so",
            "libhsa-runtime64.so.1",
            "libhsa-amd-aqlprofile64.so",
            "libhsa-amd-aqlprofile64.so.1",
            "libamd_comgr.so",
            "libamd_comgr.so.2",
            "librocprofiler-register.so",
            "librocprofiler-register.so.0",
            "libMIOpen.so",
            "libMIOpen.so.1",
            "librccl.so",
            "librccl.so.1",
            "librocblas.so",
            "librocblas.so.4",
            "libroctracer64.so",
            "libroctracer64.so.4",
            "libroctx64.so",
            "libroctx64.so.4",
            "libhipblaslt.so",
            "libhipblaslt.so.0",
            "libamdhip64.so",
            "libamdhip64.so.6",
            "libhiprtc.so",
            "libhiprtc.so.6",
            NULL,
            NULL,
        };
        for (int i = 0; replacements[i] != NULL; i += 2)
        {
            if (strcmp(filename, replacements[i]) == 0)
            {
                filename = replacements[i + 1];
                break;
            }
        }
    }
    return dlopen(filename, flags);
}
