/* File : example.i */
%module sentencepiece
%include <typemaps.i>
%include <std_vector.i>

%{
#include <stdint.h>
#include <string>
#include <vector>
#include <sentencepiece_processor.h>
#include "ffi/zig_slice.h"
%}

%insert("cheader") %{
#include "ffi/zig_slice.h"
%}

%typemap(in) absl::string_view {
    $1 = absl::string_view((char *)$input.ptr, $input.len);
}

%typemap(out, optimal="1") const std::string& %{
    $result.ptr = (void *)($1->data());
    $result.len = (size_t)($1->length());
%}
%typemap(ctype) absl::string_view, const std::string& "zig_slice"

%rename(std_string) std::string;
namespace std {
    class string {
    public:
        void reserve(size_t n);
        void clear();
        size_t capacity() const;
    };
}

%extend std::string {
    const std::string& data() const {
        return *$self;
    };
}

%extend std::vector {
    T* data() {
        return $self->data();
    };
}

%template(std_vector_int) std::vector<int>;

%typemap(out) sentencepiece::util::Status %{
    $result = $1.code();
%}
%typemap(ctype) sentencepiece::util::Status "unsigned int"
%rename(sentencepiece_util_StatusCode, fullname=1) sentencepiece::util::StatusCode;

namespace sentencepiece {
    namespace util {
        enum class StatusCode : int {
            kOk = 0,
            kCancelled = 1,
            kUnknown = 2,
            kInvalidArgument = 3,
            kDeadlineExceeded = 4,
            kNotFound = 5,
            kAlreadyExists = 6,
            kPermissionDenied = 7,
            kResourceExhausted = 8,
            kFailedPrecondition = 9,
            kAborted = 10,
            kOutOfRange = 11,
            kUnimplemented = 12,
            kInternal = 13,
            kUnavailable = 14,
            kDataLoss = 15,
            kUnauthenticated = 16,
        };
    }

    class SentencePieceProcessor {
    public:
        virtual sentencepiece::util::Status Load(absl::string_view filename);
        virtual sentencepiece::util::Status Encode(absl::string_view input, std::vector<int> *ids) const;
        virtual sentencepiece::util::Status Decode(const std::vector<int> &ids, std::string *detokenized) const;
        virtual bool IsUnknown(int id) const;
        virtual bool IsControl(int id) const;
        virtual bool IsUnused(int id) const;
        virtual bool IsByte(int id) const;
        virtual int unk_id() const;
        virtual int bos_id() const;
        virtual int eos_id() const;
        virtual int pad_id() const;
    };
}
