#include <string>
#include <iostream>
#include <dlfcn.h>
#include <errno.h>
#include <fstream>
#include <stdlib.h>
#include "tools/cpp/runfiles/runfiles.h"

namespace zml
{
    using bazel::tools::cpp::runfiles::Runfiles;

    std::unique_ptr<Runfiles> runfiles;
    std::string ROCBLAS_TENSILE_LIBPATH;
    std::string HIPBLASLT_TENSILE_LIBPATH;
    std::string HIPBLASLT_EXT_OP_LIBRARY_PATH;
    std::string ROCM_PATH;

    typedef void *(*dlopen_func)(const char *filename, int flags);
    dlopen_func dlopen_orig = nullptr;

    __attribute__((constructor)) static void setup(int argc, char **argv)
    {
        runfiles = std::unique_ptr<Runfiles>(Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY));

        HIPBLASLT_EXT_OP_LIBRARY_PATH = runfiles->Rlocation("hipblaslt-dev/lib/hipblaslt/library/hipblasltExtOpLibrary.dat");
        if (HIPBLASLT_EXT_OP_LIBRARY_PATH != "")
        {
            setenv("HIPBLASLT_EXT_OP_LIBRARY_PATH", HIPBLASLT_EXT_OP_LIBRARY_PATH.c_str(), 1);
        }

        HIPBLASLT_TENSILE_LIBPATH = runfiles->Rlocation("hipblaslt-dev/lib/hipblaslt/library");
        if (HIPBLASLT_TENSILE_LIBPATH != "")
        {
            setenv("HIPBLASLT_TENSILE_LIBPATH", HIPBLASLT_TENSILE_LIBPATH.c_str(), 1);
        }

        ROCBLAS_TENSILE_LIBPATH = runfiles->Rlocation("rocblas/lib/rocblas/library");
        setenv("ROCBLAS_TENSILE_LIBPATH", ROCBLAS_TENSILE_LIBPATH.c_str(), 1);

        ROCM_PATH = runfiles->Rlocation("libpjrt_rocm/sandbox");
        setenv("ROCM_PATH", ROCM_PATH.c_str(), 1);
    }

    static void *rocm_dlopen(const char *filename, int flags)
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
        return dlopen_orig(filename, flags);
    }
}

extern "C"
{
    zml::dlopen_func _zml_rocm_resolve_dlopen()
    {
        zml::dlopen_orig = (zml::dlopen_func)dlsym(RTLD_NEXT, "dlopen");
        return zml::rocm_dlopen;
    }

    extern void *dlopen(const char *filename, int flags) __attribute__((ifunc("_zml_rocm_resolve_dlopen")));
}
