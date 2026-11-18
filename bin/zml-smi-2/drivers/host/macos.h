#include <sys/queue.h>
// Aro cannot parse the diagnostic pragmas in the SDK's queue macros.
#undef __NULLABILITY_COMPLETENESS_PUSH
#undef __NULLABILITY_COMPLETENESS_POP
#define __NULLABILITY_COMPLETENESS_PUSH
#define __NULLABILITY_COMPLETENESS_POP
#include <libproc.h>
#include <mach/mach_host.h>
#include <stdlib.h>
enum { ZML_HOST_VM_INFO64_COUNT = HOST_VM_INFO64_REV1_COUNT };
