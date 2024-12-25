#[repr(C)]
struct ZigSlice<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> ZigSlice<T> {
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[no_mangle]
extern "C" fn hf_tokenizers_new(path: ZigSlice<u8>) -> *mut tokenizers::Tokenizer {
    return Box::into_raw(Box::new(
        tokenizers::Tokenizer::from_file(std::path::Path::new(
            std::str::from_utf8(path.as_slice()).unwrap(),
        ))
        .unwrap()
        .into(),
    ));
}

#[no_mangle]
extern "C" fn hf_tokenizers_drop(t: *mut tokenizers::Tokenizer) {
    drop(unsafe { Box::from_raw(t) });
}

#[no_mangle]
extern "C" fn hf_tokenizers_encode(
    t: *mut tokenizers::Tokenizer,
    string: ZigSlice<u8>,
) -> ZigSlice<u32> {
    let input_str = std::str::from_utf8(string.as_slice()).unwrap();
    let encoded = unsafe { t.as_ref() }
        .unwrap()
        .encode_fast(input_str, false)
        .unwrap();
    let mut ids = encoded.get_ids().to_owned().into_boxed_slice();
    let len = ids.len();
    let ret = ZigSlice {
        ptr: ids.as_mut_ptr(),
        len: len,
    };
    std::mem::forget(ids);
    return ret;
}

#[no_mangle]
extern "C" fn hf_tokenizers_tokens_drop(tokens: ZigSlice<u32>) {
    drop(unsafe { Box::from_raw(tokens.ptr) });
}

#[no_mangle]
extern "C" fn hf_tokenizers_decode(
    t: *mut tokenizers::Tokenizer,
    ids: ZigSlice<u32>,
) -> ZigSlice<u8> {
    let decoded = unsafe { t.as_ref() }
        .unwrap()
        .decode(ids.as_slice(), false)
        .unwrap();
    let len = decoded.len();
    let mut dstr = decoded.into_boxed_str();
    let ret = ZigSlice {
        ptr: dstr.as_mut_ptr(),
        len: dstr.len(),
    };
    std::mem::forget(dstr);
    return ret;
}

#[no_mangle]
extern "C" fn hf_tokenizers_str_drop(tokens: ZigSlice<u8>) {
    drop(unsafe { Box::from_raw(tokens.ptr) });
}
