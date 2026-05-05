#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include "ffi/zig_allocator.h"
#include "ffi/zig_slice.h"

namespace zml::tools {
bool DumpXSpaceFileToTraceJsonFile(std::string_view input_path,
                                   std::string_view output_path,
                                   zig_allocator* error_allocator,
                                   zig_slice* error);
}

void PrintUsage(std::string_view argv0) {
  std::cerr << "Usage: " << argv0 << " <input_xspace.pb> [output_trace.json]\n";
}

void* Alloc(const void*, size_t elem, size_t nelems, size_t) {
  return std::malloc(elem * nelems);
}

void Free(const void*, void* ptr, size_t, size_t, size_t) {
  std::free(ptr);
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string input_path = argv[1];
  const std::string output_path = argc == 3 ? argv[2] : input_path + ".trace.json";

  zig_allocator error_allocator = {
      .ctx = nullptr,
      .alloc = &Alloc,
      .free = &Free,
  };
  zig_slice error = {.ptr = nullptr, .len = 0};
  if (!zml::tools::DumpXSpaceFileToTraceJsonFile(input_path, output_path,
                                                 &error_allocator, &error)) {
    std::cerr << std::string_view(static_cast<const char*>(error.ptr), error.len)
              << "\n";
    if (error.ptr != nullptr) {
      error_allocator.deallocate(static_cast<char*>(error.ptr), error.len);
    }
    return 1;
  }

  std::cout << output_path << "\n";
  return 0;
}
