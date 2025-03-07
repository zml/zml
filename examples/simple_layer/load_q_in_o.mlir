module {
  tt.func public @_paged_attn_decode_v1_w_dot_kernel_tt_load_only(%ptr1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr3: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr4: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr5: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr6: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr7: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr8: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr9: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr10: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr11: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr12: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr13: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr14: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr15: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr16: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {

    %0 = tt.load %ptr6 : !tt.ptr<f32>
    %1 = tt.load %ptr7 : !tt.ptr<f32>  
    %2 = tt.load %ptr8 : !tt.ptr<f32>
    %3 = tt.load %ptr9 : !tt.ptr<i32>
    %4 = tt.load %ptr10 : !tt.ptr<i32>
    %5 = tt.load %ptr11 : !tt.ptr<i32>
    %6 = tt.load %ptr12 : !tt.ptr<i32>
    %7 = tt.load %ptr13 : !tt.ptr<i32>
    %8 = tt.load %ptr14 : !tt.ptr<i32>
    %9 = tt.load %ptr15 : !tt.ptr<i32>
    %10 = tt.load %ptr16 : !tt.ptr<i32>
    tt.call @_paged_attn_decode_v1_w_dot_kernel(%ptr0, %ptr1, %ptr2, %ptr3, %ptr4, %ptr5, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10) : (!tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<i32>, !tt.ptr<i64>, f32, f32, f32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    tt.return
  }

    tt.func private @_paged_attn_decode_v1_w_dot_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg5: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg6: f32 loc("/workspace/pa_decode_rework.py":159:0), %arg7: f32 loc("/workspace/pa_decode_rework.py":159:0), %arg8: f32 loc("/workspace/pa_decode_rework.py":159:0), %arg9: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg10: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg11: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg12: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg13: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg14: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg15: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":159:0), %arg16: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":159:0)) attributes {noinline = false} {
    %0 = tt.load %arg1 : !tt.ptr<bf16>
    tt.store %arg0, %0 : !tt.ptr<bf16>
    tt.return
  }
}
