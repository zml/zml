#[repr(C)]
struct ZigSlice<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> ZigSlice<T> {
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_slice_mut(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[no_mangle]
extern "C" fn hftokenizers_new(path: ZigSlice<u8>) -> *mut tokenizers::Tokenizer {
    return Box::into_raw(Box::new(
        tokenizers::Tokenizer::from_file(std::path::Path::new(
            std::str::from_utf8(path.as_slice()).unwrap(),
        ))
        .unwrap(),
    ));
}

#[no_mangle]
extern "C" fn hftokenizers_new_from_bytes(bytes: ZigSlice<u8>) -> *mut tokenizers::Tokenizer {
    return Box::into_raw(Box::new(
        tokenizers::Tokenizer::from_bytes(bytes.as_slice())
        .unwrap(),
    ));
}

#[no_mangle]
extern "C" fn hftokenizers_drop(t: *mut tokenizers::Tokenizer) {
    drop(unsafe { Box::from_raw(t) });
}

#[no_mangle]
extern "C" fn hftokenizers_encode(
    t: *mut tokenizers::Tokenizer,
    string: ZigSlice<u8>,
) -> ZigSlice<u32> {
    let input_str = std::str::from_utf8(string.as_slice()).unwrap();
    let encoded = unsafe { t.as_ref() }
        .unwrap()
        .encode_fast(input_str, false)
        .unwrap();

    // Convert the result to a boxed slice
    let mut ids: Box<[u32]> = encoded.get_ids().to_owned().into_boxed_slice();

    // Retrieve the zig slice associated to the boxed slice.
    let slice = ZigSlice {
        ptr: ids.as_mut_ptr(),
        len: ids.len(),
    };

    // Leak the box so that it's not deallocated.
    Box::leak(ids);

    return slice;
}

#[no_mangle]
extern "C" fn hftokenizers_tokens_drop(tokens: ZigSlice<u32>) {
    // Reconstruct the Box from the zig slice so that it's dropped.
    drop(unsafe { Box::from_raw(tokens.as_slice_mut()) });
}

#[no_mangle]
extern "C" fn hftokenizers_decode(
    t: *mut tokenizers::Tokenizer,
    ids: ZigSlice<u32>,
) -> ZigSlice<u8> {
    let decoded = unsafe { t.as_ref() }
        .unwrap()
        .decode(ids.as_slice(), false)
        .unwrap();

    // Convert the result to a boxed slice
    let mut string: Box<[u8]> = decoded.into_bytes().into_boxed_slice();

    // Retrieve the zig slice associated to the boxed slice.
    let slice = ZigSlice {
        ptr: string.as_mut_ptr(),
        len: string.len(),
    };

    // Leak the box so that it's not deallocated.
    Box::leak(string);

    return slice;
}

#[no_mangle]
extern "C" fn hftokenizers_str_drop(tokens: ZigSlice<u8>) {
    drop(unsafe { Box::from_raw(tokens.as_slice_mut()) });
}

#[no_mangle]
extern "C" fn hftokenizers_token_to_id(t: *mut tokenizers::Tokenizer, token: ZigSlice<u8>) -> u32 {
    let id = unsafe { t.as_ref() }
        .unwrap()
        .token_to_id(std::str::from_utf8(token.as_slice()).unwrap())
        .unwrap_or(u32::MAX);
    return id;
}
