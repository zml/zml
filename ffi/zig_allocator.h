#pragma once

#include <stdlib.h>

typedef struct
{
    const void *ctx;
    void *(*alloc)(const void *ctx, size_t elem, size_t nelems, size_t alignment);
    void (*free)(const void *ctx, void *ptr, size_t elem, size_t nelems, size_t alignment);

#ifdef __cplusplus
    template <typename T> [[nodiscard]] T *allocate(size_t n)
    {
        return static_cast<T *>(this->alloc(this->ctx, sizeof(T), n, _Alignof(T)));
    }

    template <typename T> [[nodiscard]] void deallocate(T *p, size_t n)
    {
        this->free(this->ctx, static_cast<void *>(p), sizeof(T), n, _Alignof(T));
    }
#endif
} zig_allocator;
