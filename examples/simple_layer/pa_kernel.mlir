module {
  tt.func public @paged_attention_block_h_16_pages_per_compute_block_8_batched(%arg0: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 32 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 32 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 32 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
    %0 = tt.get_program_id x : i32
    %c8_i32 = arith.constant 8 : i32
    %1 = arith.remui %0, %c8_i32 : i32
    %2 = arith.divui %0, %c8_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %3 = arith.remui %2, %c1_i32 : i32
    %4 = arith.divui %2, %c1_i32 : i32
    %5 = tt.get_program_id y : i32
    %6 = tt.get_program_id z : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %7 = arith.muli %3, %c16_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %8 = arith.muli %c0_i32, %c128_i32 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %9 = arith.muli %c0_i32_0, %c32768_i32 : i32
    %c16_i32_3 = arith.constant 16 : i32
    %10 = arith.muli %c0_i32_1, %c16_i32_3 : i32
    %c128_i32_4 = arith.constant 128 : i32
    %11 = arith.muli %c0_i32_2, %c128_i32_4 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %c32768_i32_8 = arith.constant 32768 : i32
    %12 = arith.muli %c0_i32_5, %c32768_i32_8 : i32
    %c16_i32_9 = arith.constant 16 : i32
    %13 = arith.muli %c0_i32_6, %c16_i32_9 : i32
    %c128_i32_10 = arith.constant 128 : i32
    %14 = arith.muli %c0_i32_7, %c128_i32_10 : i32
    %c0_i32_11 = arith.constant 0 : i32
    %c16_i32_12 = arith.constant 16 : i32
    %15 = arith.muli %c0_i32_11, %c16_i32_12 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %c1_i32_14 = arith.constant 1 : i32
    %16 = arith.muli %c0_i32_13, %c1_i32_14 : i32
    %c0_i32_15 = arith.constant 0 : i32
    %c16_i32_16 = arith.constant 16 : i32
    %17 = arith.muli %3, %c16_i32_16 : i32
    %c128_i32_17 = arith.constant 128 : i32
    %18 = arith.muli %c0_i32_15, %c128_i32_17 : i32
    %c16_i32_18 = arith.constant 16 : i32
    %19 = arith.muli %3, %c16_i32_18 : i32
    %c16_i32_19 = arith.constant 16 : i32
    %20 = arith.muli %3, %c16_i32_19 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %21 = tt.splat %cst : f32 -> tensor<16xf32>
    %cst_20 = arith.constant -3.40282347E+38 : f32
    %22 = tt.splat %cst_20 : f32 -> tensor<16xf32>
    %23 = arith.addf %21, %22 : tensor<16xf32>
    %cst_21 = arith.constant 0.000000e+00 : f32
    %24 = tt.splat %cst_21 : f32 -> tensor<16xf32>
    %cst_22 = arith.constant 0.000000e+00 : f32
    %25 = tt.splat %cst_22 : f32 -> tensor<16x128xf32>
    %c16_i32_23 = arith.constant 16 : i32
    %26 = arith.muli %5, %c16_i32_23 : i32
    %c16_i32_24 = arith.constant 16 : i32
    %27 = arith.addi %26, %c16_i32_24 : i32
    %c0_i32_25 = arith.constant 0 : i32
    %c0_i32_26 = arith.constant 0 : i32
    %28 = arith.addi %c0_i32_26, %6 : i32
    %c1_i32_27 = arith.constant 1 : i32
    %29 = arith.muli %28, %c1_i32_27 : i32
    %30 = arith.addi %c0_i32_25, %29 : i32
    %c0_i32_28 = arith.constant 0 : i32
    %31 = arith.addi %c0_i32_28, %16 : i32
    %c1_i32_29 = arith.constant 1 : i32
    %32 = arith.muli %31, %c1_i32_29 : i32
    %33 = arith.addi %30, %32 : i32
    %34 = tt.addptr %arg4, %33 : !tt.ptr<i32>, i32
    %35 = tt.load %34 : !tt.ptr<i32>
    %c16_i32_30 = arith.constant 16 : i32
    %36 = arith.addi %35, %c16_i32_30 : i32
    %c1_i32_31 = arith.constant 1 : i32
    %37 = arith.subi %36, %c1_i32_31 : i32
    %c16_i32_32 = arith.constant 16 : i32
    %38 = arith.divsi %37, %c16_i32_32 : i32
    %39 = arith.minsi %38, %27 : i32
    %40 = arith.cmpi sge, %26, %39 : i32
    %41 = arith.extui %40 : i1 to i32
    %c0_i32_33 = arith.constant 0 : i32
    %42 = arith.cmpi eq, %41, %c0_i32_33 : i32
    %43:3 = scf.if %42 -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>) {
      %c0_i32_55 = arith.constant 0 : i32
      %139 = tt.splat %c0_i32_55 : i32 -> tensor<16x128xi32>
      %c0_i32_56 = arith.constant 0 : i32
      %140 = tt.splat %c0_i32_56 : i32 -> tensor<16x128xi32>
      %141 = tt.splat %6 : i32 -> tensor<16x128xi32>
      %142 = arith.addi %140, %141 : tensor<16x128xi32>
      %c16384_i32_57 = arith.constant 16384 : i32
      %143 = tt.splat %c16384_i32_57 : i32 -> tensor<16x128xi32>
      %144 = arith.muli %142, %143 : tensor<16x128xi32>
      %145 = arith.addi %139, %144 : tensor<16x128xi32>
      %c0_i32_58 = arith.constant 0 : i32
      %146 = tt.splat %c0_i32_58 : i32 -> tensor<16x128xi32>
      %147 = tt.splat %1 : i32 -> tensor<16x128xi32>
      %148 = arith.addi %146, %147 : tensor<16x128xi32>
      %c2048_i32_59 = arith.constant 2048 : i32
      %149 = tt.splat %c2048_i32_59 : i32 -> tensor<16x128xi32>
      %150 = arith.muli %148, %149 : tensor<16x128xi32>
      %151 = arith.addi %145, %150 : tensor<16x128xi32>
      %152 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %153 = tt.expand_dims %152 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %154 = tt.broadcast %153 : tensor<16x1xi32> -> tensor<16x128xi32>
      %155 = tt.splat %7 : i32 -> tensor<16x128xi32>
      %156 = arith.addi %154, %155 : tensor<16x128xi32>
      %c128_i32_60 = arith.constant 128 : i32
      %157 = tt.splat %c128_i32_60 : i32 -> tensor<16x128xi32>
      %158 = arith.muli %156, %157 : tensor<16x128xi32>
      %159 = arith.addi %151, %158 : tensor<16x128xi32>
      %160 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %161 = tt.expand_dims %160 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %162 = tt.broadcast %161 : tensor<1x128xi32> -> tensor<16x128xi32>
      %163 = tt.splat %8 : i32 -> tensor<16x128xi32>
      %164 = arith.addi %162, %163 : tensor<16x128xi32>
      %c1_i32_61 = arith.constant 1 : i32
      %165 = tt.splat %c1_i32_61 : i32 -> tensor<16x128xi32>
      %166 = arith.muli %164, %165 : tensor<16x128xi32>
      %167 = arith.addi %159, %166 : tensor<16x128xi32>
      %168 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>>
      %169 = tt.addptr %168, %167 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
      %170 = tt.load %169 : tensor<16x128x!tt.ptr<bf16>>
      %171 = arith.subi %39, %26 : i32
      %c8_i32_62 = arith.constant 8 : i32 
      %172 = arith.addi %171, %c8_i32_62 : i32 
      %c1_i32_63 = arith.constant 1 : i32
      %173 = arith.subi %172, %c1_i32_63 : i32
      %c8_i32_64 = arith.constant 8 : i32
      %174 = arith.divsi %173, %c8_i32_64 : i32
      %c0_i32_65 = arith.constant 0 : i32 
      %c1_i32_66 = arith.constant 1 : i32 
      %175:3 = scf.for %arg8 = %c0_i32_65 to %174 step %c1_i32_66 iter_args(%arg9 = %25, %arg10 = %23, %arg11 = %24) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)  : i32 {
        %c8_i32_67 = arith.constant 8 : i32
        %176 = arith.muli %arg8, %c8_i32_67 : i32
        %c0_i32_68 = arith.constant 0 : i32
        %177 = tt.splat %c0_i32_68 : i32 -> tensor<8xi32>
        %c0_i32_69 = arith.constant 0 : i32
        %178 = tt.splat %c0_i32_69 : i32 -> tensor<8xi32>
        %179 = tt.splat %6 : i32 -> tensor<8xi32>
        %180 = arith.addi %178, %179 : tensor<8xi32>
        %c128_i32_70 = arith.constant 128 : i32
        %181 = tt.splat %c128_i32_70 : i32 -> tensor<8xi32>
        %182 = arith.muli %180, %181 : tensor<8xi32>
        %183 = arith.addi %177, %182 : tensor<8xi32>
        %c0_i32_71 = arith.constant 0 : i32
        %184 = tt.splat %c0_i32_71 : i32 -> tensor<8xi32>
        %185 = tt.splat %5 : i32 -> tensor<8xi32>
        %186 = arith.addi %184, %185 : tensor<8xi32>
        %c16_i32_72 = arith.constant 16 : i32
        %187 = tt.splat %c16_i32_72 : i32 -> tensor<8xi32>
        %188 = arith.muli %186, %187 : tensor<8xi32>
        %189 = arith.addi %183, %188 : tensor<8xi32>
        %190 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
        %191 = tt.splat %176 : i32 -> tensor<8xi32>
        %192 = arith.addi %191, %190 : tensor<8xi32>
        %193 = tt.splat %15 : i32 -> tensor<8xi32>
        %194 = arith.addi %192, %193 : tensor<8xi32>
        %c1_i32_73 = arith.constant 1 : i32
        %195 = tt.splat %c1_i32_73 : i32 -> tensor<8xi32>
        %196 = arith.muli %194, %195 : tensor<8xi32>
        %197 = arith.addi %189, %196 : tensor<8xi32>
        %198 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
        %199 = tt.addptr %198, %197 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
        %200 = tt.load %199 : tensor<8x!tt.ptr<i32>>
        %c0_i32_74 = arith.constant 0 : i32
        %201 = tt.splat %c0_i32_74 : i32 -> tensor<8x16x128xi32>
        %c0_i32_75 = arith.constant 0 : i32
        %202 = tt.splat %c0_i32_75 : i32 -> tensor<8x16x128xi32>
        %203 = tt.splat %1 : i32 -> tensor<8x16x128xi32>
        %204 = arith.addi %202, %203 : tensor<8x16x128xi32>
        %c67108864_i32 = arith.constant 67108864 : i32
        %205 = tt.splat %c67108864_i32 : i32 -> tensor<8x16x128xi32>
        %206 = arith.muli %204, %205 : tensor<8x16x128xi32>
        %207 = arith.addi %201, %206 : tensor<8x16x128xi32>
        %208 = tt.expand_dims %200 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
        %209 = tt.expand_dims %208 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
        %210 = tt.broadcast %209 : tensor<8x1x1xi32> -> tensor<8x16x128xi32>
        %211 = tt.splat %9 : i32 -> tensor<8x16x128xi32>
        %212 = arith.addi %210, %211 : tensor<8x16x128xi32>
        %c2048_i32_76 = arith.constant 2048 : i32
        %213 = tt.splat %c2048_i32_76 : i32 -> tensor<8x16x128xi32>
        %214 = arith.muli %212, %213 : tensor<8x16x128xi32>
        %215 = arith.addi %207, %214 : tensor<8x16x128xi32>
        %216 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %217 = tt.expand_dims %216 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
        %218 = tt.expand_dims %217 {axis = 0 : i32} : tensor<16x1xi32> -> tensor<1x16x1xi32>
        %219 = tt.broadcast %218 : tensor<1x16x1xi32> -> tensor<8x16x128xi32>
        %220 = tt.splat %10 : i32 -> tensor<8x16x128xi32>
        %221 = arith.addi %219, %220 : tensor<8x16x128xi32>
        %c128_i32_77 = arith.constant 128 : i32
        %222 = tt.splat %c128_i32_77 : i32 -> tensor<8x16x128xi32>
        %223 = arith.muli %221, %222 : tensor<8x16x128xi32>
        %224 = arith.addi %215, %223 : tensor<8x16x128xi32>
        %225 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %226 = tt.expand_dims %225 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %227 = tt.expand_dims %226 {axis = 0 : i32} : tensor<1x128xi32> -> tensor<1x1x128xi32>
        %228 = tt.broadcast %227 : tensor<1x1x128xi32> -> tensor<8x16x128xi32>
        %229 = tt.splat %11 : i32 -> tensor<8x16x128xi32>
        %230 = arith.addi %228, %229 : tensor<8x16x128xi32>
        %c1_i32_78 = arith.constant 1 : i32
        %231 = tt.splat %c1_i32_78 : i32 -> tensor<8x16x128xi32>
        %232 = arith.muli %230, %231 : tensor<8x16x128xi32>
        %233 = arith.addi %224, %232 : tensor<8x16x128xi32>
        %234 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x16x128x!tt.ptr<bf16>>
        %235 = tt.addptr %234, %233 : tensor<8x16x128x!tt.ptr<bf16>>, tensor<8x16x128xi32>
        %236 = tt.load %235 : tensor<8x16x128x!tt.ptr<bf16>>
        %237 = tt.reshape %236 : tensor<8x16x128xbf16> -> tensor<128x128xbf16>
        %c0_i32_79 = arith.constant 0 : i32
        %238 = tt.splat %c0_i32_79 : i32 -> tensor<8x16x128xi32>
        %c0_i32_80 = arith.constant 0 : i32
        %239 = tt.splat %c0_i32_80 : i32 -> tensor<8x16x128xi32>
        %240 = tt.splat %1 : i32 -> tensor<8x16x128xi32>
        %241 = arith.addi %239, %240 : tensor<8x16x128xi32>
        %c67108864_i32_81 = arith.constant 67108864 : i32
        %242 = tt.splat %c67108864_i32_81 : i32 -> tensor<8x16x128xi32>
        %243 = arith.muli %241, %242 : tensor<8x16x128xi32>
        %244 = arith.addi %238, %243 : tensor<8x16x128xi32>
        %245 = tt.expand_dims %200 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
        %246 = tt.expand_dims %245 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
        %247 = tt.broadcast %246 : tensor<8x1x1xi32> -> tensor<8x16x128xi32>
        %248 = tt.splat %12 : i32 -> tensor<8x16x128xi32>
        %249 = arith.addi %247, %248 : tensor<8x16x128xi32>
        %c2048_i32_82 = arith.constant 2048 : i32
        %250 = tt.splat %c2048_i32_82 : i32 -> tensor<8x16x128xi32>
        %251 = arith.muli %249, %250 : tensor<8x16x128xi32>
        %252 = arith.addi %244, %251 : tensor<8x16x128xi32>
        %253 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %254 = tt.expand_dims %253 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
        %255 = tt.expand_dims %254 {axis = 0 : i32} : tensor<16x1xi32> -> tensor<1x16x1xi32>
        %256 = tt.broadcast %255 : tensor<1x16x1xi32> -> tensor<8x16x128xi32>
        %257 = tt.splat %13 : i32 -> tensor<8x16x128xi32>
        %258 = arith.addi %256, %257 : tensor<8x16x128xi32>
        %c128_i32_83 = arith.constant 128 : i32
        %259 = tt.splat %c128_i32_83 : i32 -> tensor<8x16x128xi32>
        %260 = arith.muli %258, %259 : tensor<8x16x128xi32>
        %261 = arith.addi %252, %260 : tensor<8x16x128xi32>
        %262 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %263 = tt.expand_dims %262 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %264 = tt.expand_dims %263 {axis = 0 : i32} : tensor<1x128xi32> -> tensor<1x1x128xi32>
        %265 = tt.broadcast %264 : tensor<1x1x128xi32> -> tensor<8x16x128xi32>
        %266 = tt.splat %14 : i32 -> tensor<8x16x128xi32>
        %267 = arith.addi %265, %266 : tensor<8x16x128xi32>
        %c1_i32_84 = arith.constant 1 : i32
        %268 = tt.splat %c1_i32_84 : i32 -> tensor<8x16x128xi32>
        %269 = arith.muli %267, %268 : tensor<8x16x128xi32>
        %270 = arith.addi %261, %269 : tensor<8x16x128xi32>
        %271 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<8x16x128x!tt.ptr<bf16>>
        %272 = tt.addptr %271, %270 : tensor<8x16x128x!tt.ptr<bf16>>, tensor<8x16x128xi32>
        %273 = tt.load %272 : tensor<8x16x128x!tt.ptr<bf16>>
        %274 = tt.reshape %273 : tensor<8x16x128xbf16> -> tensor<128x128xbf16>
        %275 = tt.trans %237 {order = array<i32: 1, 0>} : tensor<128x128xbf16> -> tensor<128x128xbf16>
        %cst_85 = arith.constant 0.000000e+00 : f32 
        %276 = tt.splat %cst_85 : f32 -> tensor<16x128xf32> 
        %277 = tt.dot %170, %275, %276, inputPrecision = tf32 : tensor<16x128xbf16> * tensor<128x128xbf16> -> tensor<16x128xf32> 
        %c16_i32_86 = arith.constant 16 : i32
        %278 = arith.muli %5, %c16_i32_86 : i32
        %c8_i32_87 = arith.constant 8 : i32
        %279 = arith.muli %arg8, %c8_i32_87 : i32
        %280 = arith.addi %278, %279 : i32
        %c16_i32_88 = arith.constant 16 : i32
        %281 = arith.muli %280, %c16_i32_88 : i32
        %282 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %283 = tt.splat %281 : i32 -> tensor<128xi32>
        %284 = arith.addi %282, %283 : tensor<128xi32>
        %c0_i32_89 = arith.constant 0 : i32
        %c0_i32_90 = arith.constant 0 : i32
        %285 = arith.addi %c0_i32_90, %6 : i32
        %c1_i32_91 = arith.constant 1 : i32
        %286 = arith.muli %285, %c1_i32_91 : i32
        %287 = arith.addi %c0_i32_89, %286 : i32
        %c0_i32_92 = arith.constant 0 : i32
        %288 = arith.addi %c0_i32_92, %16 : i32
        %c1_i32_93 = arith.constant 1 : i32
        %289 = arith.muli %288, %c1_i32_93 : i32
        %290 = arith.addi %287, %289 : i32
        %291 = tt.addptr %arg4, %290 : !tt.ptr<i32>, i32
        %292 = tt.load %291 : !tt.ptr<i32>
        %293 = tt.splat %292 : i32 -> tensor<128xi32>
        %294 = arith.cmpi slt, %284, %293 : tensor<128xi32>
        %295 = tt.expand_dims %294 {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1> 
        %296 = tt.broadcast %295 : tensor<1x128xi1> -> tensor<16x128xi1> 
        %cst_94 = arith.constant -2.38197633E+38 : f32
        %297 = tt.splat %cst_94 : f32 -> tensor<16x128xf32> 
        %298 = arith.select %296, %277, %297 : tensor<16x128xi1>, tensor<16x128xf32> 
        %299 = "tt.reduce"(%298) <{axis = 1 : i32}> ({
        ^bb0(%arg12: f32, %arg13: f32):
          %321 = arith.maxnumf %arg12, %arg13 : f32
          tt.reduce.return %321 : f32
        }) : (tensor<16x128xf32>) -> tensor<16xf32>
        %300 = arith.maxnumf %arg10, %299 : tensor<16xf32>
        %301 = arith.subf %arg10, %300 : tensor<16xf32>
        %cst_95 = arith.constant 1.44269502 : f32
        %302 = tt.splat %cst_95 : f32 -> tensor<16xf32>
        %303 = arith.mulf %301, %302 : tensor<16xf32>
        %304 = tt.extern_elementwise %303 {libname = "", libpath = "", pure = true, symbol = "__nv_exp2f"} : (tensor<16xf32>) -> tensor<16xf32>
        %305 = arith.mulf %304, %arg11 : tensor<16xf32>
        %306 = tt.expand_dims %300 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %307 = tt.broadcast %306 : tensor<16x1xf32> -> tensor<16x128xf32>
        %308 = arith.subf %298, %307 : tensor<16x128xf32>
        %cst_96 = arith.constant 1.44269502 : f32
        %309 = tt.splat %cst_96 : f32 -> tensor<16x128xf32>
        %310 = arith.mulf %308, %309 : tensor<16x128xf32>
        %311 = tt.extern_elementwise %310 {libname = "", libpath = "", pure = true, symbol = "__nv_exp2f"} : (tensor<16x128xf32>) -> tensor<16x128xf32> 
        %312 = "tt.reduce"(%311) <{axis = 1 : i32}> ({
        ^bb0(%arg12: f32, %arg13: f32):
          %321 = arith.addf %arg12, %arg13 : f32
          tt.reduce.return %321 : f32
        }) : (tensor<16x128xf32>) -> tensor<16xf32>
        %313 = arith.addf %305, %312 : tensor<16xf32>
        %314 = tt.expand_dims %304 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
        %315 = tt.broadcast %314 : tensor<16x1xf32> -> tensor<16x128xf32>
        %316 = arith.mulf %315, %arg9 : tensor<16x128xf32>
        %317 = arith.truncf %311 : tensor<16x128xf32> to tensor<16x128xbf16>
        %cst_97 = arith.constant 0.000000e+00 : f32
        %318 = tt.splat %cst_97 : f32 -> tensor<16x128xf32>
        %319 = tt.dot %317, %274, %318, inputPrecision = tf32 : tensor<16x128xbf16> * tensor<128x128xbf16> -> tensor<16x128xf32>
        %320 = arith.addf %316, %319 : tensor<16x128xf32>
        scf.yield %320, %300, %313 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32> 
      } 
      scf.yield %175#0, %175#1, %175#2 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
    } else {
      scf.yield %25, %23, %24 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>
    }
    %44 = arith.truncf %43#0 : tensor<16x128xf32> to tensor<16x128xbf16>
    %c0_i32_34 = arith.constant 0 : i32
    %45 = tt.splat %c0_i32_34 : i32 -> tensor<16x128xi32>
    %c0_i32_35 = arith.constant 0 : i32
    %46 = tt.splat %c0_i32_35 : i32 -> tensor<16x128xi32>
    %47 = tt.splat %6 : i32 -> tensor<16x128xi32>
    %48 = arith.addi %46, %47 : tensor<16x128xi32>
    %c131072_i32 = arith.constant 131072 : i32
    %49 = tt.splat %c131072_i32 : i32 -> tensor<16x128xi32>
    %50 = arith.muli %48, %49 : tensor<16x128xi32>
    %51 = arith.addi %45, %50 : tensor<16x128xi32>
    %c0_i32_36 = arith.constant 0 : i32
    %52 = tt.splat %c0_i32_36 : i32 -> tensor<16x128xi32>
    %53 = tt.splat %5 : i32 -> tensor<16x128xi32>
    %54 = arith.addi %52, %53 : tensor<16x128xi32>
    %c16384_i32 = arith.constant 16384 : i32
    %55 = tt.splat %c16384_i32 : i32 -> tensor<16x128xi32>
    %56 = arith.muli %54, %55 : tensor<16x128xi32>
    %57 = arith.addi %51, %56 : tensor<16x128xi32>
    %c0_i32_37 = arith.constant 0 : i32
    %58 = tt.splat %c0_i32_37 : i32 -> tensor<16x128xi32>
    %59 = tt.splat %1 : i32 -> tensor<16x128xi32>
    %60 = arith.addi %58, %59 : tensor<16x128xi32>
    %c2048_i32 = arith.constant 2048 : i32
    %61 = tt.splat %c2048_i32 : i32 -> tensor<16x128xi32>
    %62 = arith.muli %60, %61 : tensor<16x128xi32>
    %63 = arith.addi %57, %62 : tensor<16x128xi32>
    %64 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %66 = tt.broadcast %65 : tensor<16x1xi32> -> tensor<16x128xi32>
    %67 = tt.splat %17 : i32 -> tensor<16x128xi32>
    %68 = arith.addi %66, %67 : tensor<16x128xi32>
    %c128_i32_38 = arith.constant 128 : i32
    %69 = tt.splat %c128_i32_38 : i32 -> tensor<16x128xi32>
    %70 = arith.muli %68, %69 : tensor<16x128xi32>
    %71 = arith.addi %63, %70 : tensor<16x128xi32>
    %72 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %73 = tt.expand_dims %72 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %74 = tt.broadcast %73 : tensor<1x128xi32> -> tensor<16x128xi32>
    %75 = tt.splat %18 : i32 -> tensor<16x128xi32>
    %76 = arith.addi %74, %75 : tensor<16x128xi32>
    %c1_i32_39 = arith.constant 1 : i32
    %77 = tt.splat %c1_i32_39 : i32 -> tensor<16x128xi32>
    %78 = arith.muli %76, %77 : tensor<16x128xi32>
    %79 = arith.addi %71, %78 : tensor<16x128xi32>
    %80 = tt.splat %arg5 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>>
    %81 = tt.addptr %80, %79 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32>
    %82 = tt.load %81 : tensor<16x128x!tt.ptr<bf16>>
    tt.store %81, %44 : tensor<16x128x!tt.ptr<bf16>>
    %c0_i32_40 = arith.constant 0 : i32
    %83 = tt.splat %c0_i32_40 : i32 -> tensor<16xi32>
    %c0_i32_41 = arith.constant 0 : i32
    %84 = tt.splat %c0_i32_41 : i32 -> tensor<16xi32>
    %85 = tt.splat %6 : i32 -> tensor<16xi32>
    %86 = arith.addi %84, %85 : tensor<16xi32>
    %c1024_i32 = arith.constant 1024 : i32
    %87 = tt.splat %c1024_i32 : i32 -> tensor<16xi32>
    %88 = arith.muli %86, %87 : tensor<16xi32>
    %89 = arith.addi %83, %88 : tensor<16xi32>
    %c0_i32_42 = arith.constant 0 : i32
    %90 = tt.splat %c0_i32_42 : i32 -> tensor<16xi32>
    %91 = tt.splat %5 : i32 -> tensor<16xi32>
    %92 = arith.addi %90, %91 : tensor<16xi32>
    %c128_i32_43 = arith.constant 128 : i32
    %93 = tt.splat %c128_i32_43 : i32 -> tensor<16xi32>
    %94 = arith.muli %92, %93 : tensor<16xi32>
    %95 = arith.addi %89, %94 : tensor<16xi32>
    %c0_i32_44 = arith.constant 0 : i32
    %96 = tt.splat %c0_i32_44 : i32 -> tensor<16xi32>
    %97 = tt.splat %1 : i32 -> tensor<16xi32>
    %98 = arith.addi %96, %97 : tensor<16xi32>
    %c16_i32_45 = arith.constant 16 : i32
    %99 = tt.splat %c16_i32_45 : i32 -> tensor<16xi32>
    %100 = arith.muli %98, %99 : tensor<16xi32>
    %101 = arith.addi %95, %100 : tensor<16xi32>
    %102 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %103 = tt.splat %19 : i32 -> tensor<16xi32>
    %104 = arith.addi %102, %103 : tensor<16xi32>
    %c1_i32_46 = arith.constant 1 : i32
    %105 = tt.splat %c1_i32_46 : i32 -> tensor<16xi32>
    %106 = arith.muli %104, %105 : tensor<16xi32>
    %107 = arith.addi %101, %106 : tensor<16xi32>
    %108 = tt.splat %arg6 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %109 = tt.addptr %108, %107 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %110 = tt.load %109 : tensor<16x!tt.ptr<f32>>
    tt.store %109, %43#2 : tensor<16x!tt.ptr<f32>>
    %c0_i32_47 = arith.constant 0 : i32
    %111 = tt.splat %c0_i32_47 : i32 -> tensor<16xi32>
    %c0_i32_48 = arith.constant 0 : i32
    %112 = tt.splat %c0_i32_48 : i32 -> tensor<16xi32>
    %113 = tt.splat %6 : i32 -> tensor<16xi32>
    %114 = arith.addi %112, %113 : tensor<16xi32>
    %c1024_i32_49 = arith.constant 1024 : i32
    %115 = tt.splat %c1024_i32_49 : i32 -> tensor<16xi32>
    %116 = arith.muli %114, %115 : tensor<16xi32>
    %117 = arith.addi %111, %116 : tensor<16xi32>
    %c0_i32_50 = arith.constant 0 : i32
    %118 = tt.splat %c0_i32_50 : i32 -> tensor<16xi32>
    %119 = tt.splat %5 : i32 -> tensor<16xi32>
    %120 = arith.addi %118, %119 : tensor<16xi32>
    %c128_i32_51 = arith.constant 128 : i32
    %121 = tt.splat %c128_i32_51 : i32 -> tensor<16xi32>
    %122 = arith.muli %120, %121 : tensor<16xi32>
    %123 = arith.addi %117, %122 : tensor<16xi32>
    %c0_i32_52 = arith.constant 0 : i32
    %124 = tt.splat %c0_i32_52 : i32 -> tensor<16xi32>
    %125 = tt.splat %1 : i32 -> tensor<16xi32>
    %126 = arith.addi %124, %125 : tensor<16xi32>
    %c16_i32_53 = arith.constant 16 : i32
    %127 = tt.splat %c16_i32_53 : i32 -> tensor<16xi32>
    %128 = arith.muli %126, %127 : tensor<16xi32>
    %129 = arith.addi %123, %128 : tensor<16xi32>
    %130 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %131 = tt.splat %20 : i32 -> tensor<16xi32>
    %132 = arith.addi %130, %131 : tensor<16xi32>
    %c1_i32_54 = arith.constant 1 : i32
    %133 = tt.splat %c1_i32_54 : i32 -> tensor<16xi32>
    %134 = arith.muli %132, %133 : tensor<16xi32>
    %135 = arith.addi %129, %134 : tensor<16xi32>
    %136 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %137 = tt.addptr %136, %135 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %138 = tt.load %137 : tensor<16x!tt.ptr<f32>>
    tt.store %137, %43#1 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
