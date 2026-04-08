#include "tracy_shim.h"

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

struct SourceLocationKey {
    const char* name;
    const char* function;
    const char* file;
    uint32_t line;
    uint32_t color;

    bool operator==(const SourceLocationKey& other) const {
        return name == other.name and function == other.function and file == other.file and
            line == other.line and color == other.color;
    }
};

struct SourceLocationKeyHash {
    size_t operator()(const SourceLocationKey& key) const {
        size_t seed = 0;
        hashCombine(seed, reinterpret_cast<uintptr_t>(key.name));
        hashCombine(seed, reinterpret_cast<uintptr_t>(key.function));
        hashCombine(seed, reinterpret_cast<uintptr_t>(key.file));
        hashCombine(seed, key.line);
        hashCombine(seed, key.color);
        return seed;
    }

    template <typename T>
    static void hashCombine(size_t& seed, T value) {
        seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

std::mutex g_source_location_mutex;
std::unordered_map<SourceLocationKey, uint64_t, SourceLocationKeyHash> g_source_locations;

uint64_t getSourceLocation(
    const char* name,
    const char* function,
    const char* file,
    uint32_t line,
    uint32_t color) {
    const SourceLocationKey key = {
        .name = name,
        .function = function,
        .file = file,
        .line = line,
        .color = color,
    };

    {
        std::lock_guard<std::mutex> lock(g_source_location_mutex);
        const auto it = g_source_locations.find(key);
        if (it != g_source_locations.end()) return it->second;
    }

    const char* safe_name = name != nullptr ? name : "";
    const char* safe_function = function != nullptr ? function : "";
    const char* safe_file = file != nullptr ? file : "";

    const uint64_t srcloc = safe_name[0] != '\0'
        ? ___tracy_alloc_srcloc_name(
            line,
            safe_file,
            std::strlen(safe_file),
            safe_function,
            std::strlen(safe_function),
            safe_name,
            std::strlen(safe_name),
            color)
        : ___tracy_alloc_srcloc(
            line,
            safe_file,
            std::strlen(safe_file),
            safe_function,
            std::strlen(safe_function),
            color);

    std::lock_guard<std::mutex> lock(g_source_location_mutex);
    return g_source_locations.emplace(key, srcloc).first->second;
}

}  // namespace

extern "C" void zml_tracy_set_thread_name(const char* name) {
    TracyCSetThreadName(name);
}

extern "C" int32_t zml_tracy_is_connected(void) {
    return ___tracy_connected();
}

extern "C" ZmlTracyZoneCtx zml_tracy_zone_begin(
    const char* name,
    const char* function,
    const char* file,
    uint32_t line,
    uint32_t color) {
    const uint64_t srcloc = getSourceLocation(name, function, file, line, color);
    return ___tracy_emit_zone_begin_alloc_callstack(srcloc, TRACY_CALLSTACK, 1);
}

extern "C" void zml_tracy_zone_end(ZmlTracyZoneCtx ctx) {
    TracyCZoneEnd(ctx);
}

extern "C" void zml_tracy_zone_text(ZmlTracyZoneCtx ctx, const char* text, size_t size) {
    TracyCZoneText(ctx, text, size);
}

extern "C" void zml_tracy_zone_value(ZmlTracyZoneCtx ctx, uint64_t value) {
    TracyCZoneValue(ctx, value);
}

extern "C" void zml_tracy_frame_mark(void) {
    TracyCFrameMark;
}

extern "C" void zml_tracy_frame_mark_named(const char* name) {
    TracyCFrameMarkNamed(name);
}
