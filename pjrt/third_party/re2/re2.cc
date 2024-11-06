#include "absl/strings/string_view.h"
#include "absl/strings/numbers.h"

#include "re2/re2.h"

extern "C" {
  bool isTfOpName(const char *op_name, size_t len) {
    absl::string_view op_name_view(op_name, len);
    static const LazyRE2 kTfOpNameRegEx = {"[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*"};
    return RE2::FullMatch(op_name_view, *kTfOpNameRegEx);
  }

  bool isJaxOpType(const char *op_type, size_t len) {
    absl::string_view op_type_view(op_type, len);
    // Jax op type should start with lowercase character or underscore.
    // If it contains '[]', it must end with ']' and whatever chars inside
    // it are considered as a match.
    static const LazyRE2 kJaxOpTypeRegEx = {"[a-z_][a-z0-9_]*(\\[.*\\])?"};
    return RE2::FullMatch(op_type_view, *kJaxOpTypeRegEx);
  }

  bool isNumber(const char *str, size_t len) {
    int64_t unused;
    absl::string_view str_view(str, len);
    return absl::SimpleAtoi(str_view, &unused);
  }
}

