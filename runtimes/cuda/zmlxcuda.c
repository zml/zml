#include <dlfcn.h>
#include <string.h>

void *zmlxcuda_dlopen(const char *filename, int flags)
{
    if (filename != NULL)
    {
        char *replacements[] = {
            "libcublas.so",
            "libcublas.so.12",
            "libcublasLt.so",
            "libcublasLt.so.12",
            "libcudart.so",
            "libcudart.so.12",
            "libcudnn.so",
            "libcudnn.so.9",
            "libcufft.so",
            "libcufft.so.11",
            "libcupti.so",
            "libcupti.so.12",
            "libcusolver.so",
            "libcusolver.so.11",
            "libcusparse.so",
            "libcusparse.so.12",
            "libnccl.so",
            "libnccl.so.2",
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
