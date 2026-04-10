#include <iostream>
#include <string>
#include <string_view>

namespace zml::tools {
bool DumpXSpaceFileToTraceJsonFile(std::string_view input_path,
                                   std::string_view output_path,
                                   std::string* error);
}

void PrintUsage(std::string_view argv0) {
  std::cerr << "Usage: " << argv0 << " <input_xspace.pb> [output_trace.json]\n";
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string input_path = argv[1];
  const std::string output_path = argc == 3 ? argv[2] : input_path + ".trace.json";

  std::string error;
  if (!zml::tools::DumpXSpaceFileToTraceJsonFile(input_path, output_path,
                                                 &error)) {
    std::cerr << error << "\n";
    return 1;
  }

  std::cout << output_path << "\n";
  return 0;
}
