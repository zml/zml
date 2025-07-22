#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

void *zmlxrocm_dlopen(const char *filename, int flags) __attribute__((visibility("default")))
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
            "libamd_comgr.so.3",
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
