#include <dlfcn.h>
#include <string.h>

void *zmlxneuron_dlopen(const char *filename, int flags)
{
    if (filename != NULL)
    {
        char *replacements[] = {
            "libnccom.so", "libnccom.so.2",
            "libnccom-net.so", "libnccom-net.so.0",
            "libnrt.so", "libnrt.so.1",
            "libncfw.so", "libncfw.so.2",
            NULL, NULL,
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
