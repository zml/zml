#loc = loc("/workspace/pa_decode_rework.py":157:0)
#loc1 = loc(unknown)
#loc54 = loc("/workspace/pa_decode_rework.py":271:42)
#loc71 = loc("/workspace/pa_decode_rework.py":286:43)
#loc89 = loc(callsite(#loc1 at #loc54))
#loc93 = loc(callsite(#loc1 at #loc71))
module {
  tt.func public @_paged_attn_decode_v1_w_dot_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg5: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg6: f32 loc("/workspace/pa_decode_rework.py":157:0), %arg7: f32 loc("/workspace/pa_decode_rework.py":157:0), %arg8: f32 loc("/workspace/pa_decode_rework.py":157:0), %arg9: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg10: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg11: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg12: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg13: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg14: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg15: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":157:0), %arg16: i32 {tt.divisibility = 16 : i32} loc("/workspace/pa_decode_rework.py":157:0)) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<16xf32> loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32> loc(#loc1)
    %c15_i64 = arith.constant 15 : i64 loc(#loc1)
    %c1_i64 = arith.constant 1 : i64 loc(#loc1)
    %c0_i64 = arith.constant 0 : i64 loc(#loc1)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x128xbf16> loc(#loc1)
    %cst_2 = arith.constant dense<1.44269502> : tensor<16xf32> loc(#loc1)
    %cst_3 = arith.constant dense<1.44269502> : tensor<16x16xf32> loc(#loc1)
    %cst_4 = arith.constant dense<0xFF800000> : tensor<16x16xf32> loc(#loc1)
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<16x16xf32> loc(#loc1)
    %cst_6 = arith.constant dense<16> : tensor<16x1xi32> loc(#loc1)
    %c16_i64 = arith.constant 16 : i64 loc(#loc1)
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<16x128xf32> loc(#loc1)
    %cst_8 = arith.constant dense<128> : tensor<1x128xi32> loc(#loc1)
    %cst_9 = arith.constant dense<4> : tensor<16x1xi32> loc(#loc1)
    %c4_i32 = arith.constant 4 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = tt.get_program_id y : i32 loc(#loc3)
    %2 = tt.addptr %arg5, %0 : !tt.ptr<i64>, i32 loc(#loc4)
    %3 = tt.load %2 : !tt.ptr<i64> loc(#loc5)
    %4 = arith.addi %3, %c15_i64 : i64 loc(#loc86)
    %5 = arith.divsi %4, %c16_i64 : i64 loc(#loc87)
    %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc9)
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32> loc(#loc10)
    %8 = arith.muli %0, %arg11 : i32 loc(#loc11)
    %9 = arith.muli %1, %c4_i32 : i32 loc(#loc12)
    %10 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> loc(#loc13)
    %11 = tt.splat %9 : i32 -> tensor<16x1xi32> loc(#loc14)
    %12 = arith.addi %11, %10 : tensor<16x1xi32> loc(#loc14)
    %13 = tt.splat %arg12 : i32 -> tensor<16x1xi32> loc(#loc15)
    %14 = arith.muli %12, %13 : tensor<16x1xi32> loc(#loc15)
    %15 = tt.splat %8 : i32 -> tensor<16x1xi32> loc(#loc16)
    %16 = arith.addi %15, %14 : tensor<16x1xi32> loc(#loc16)
    %17 = tt.expand_dims %7 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32> loc(#loc17)
    %18 = tt.broadcast %16 : tensor<16x1xi32> -> tensor<16x128xi32> loc(#loc18)
    %19 = tt.broadcast %17 : tensor<1x128xi32> -> tensor<16x128xi32> loc(#loc18)
    %20 = arith.addi %18, %19 : tensor<16x128xi32> loc(#loc18)
    %21 = arith.cmpi slt, %10, %cst_9 : tensor<16x1xi32> loc(#loc19)
    %22 = arith.cmpi slt, %17, %cst_8 : tensor<1x128xi32> loc(#loc20)
    %23 = tt.broadcast %21 : tensor<16x1xi1> -> tensor<16x128xi1> loc(#loc21)
    %24 = tt.broadcast %22 : tensor<1x128xi1> -> tensor<16x128xi1> loc(#loc21)
    %25 = arith.andi %23, %24 : tensor<16x128xi1> loc(#loc21)
    %26 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>> loc(#loc22)
    %27 = tt.addptr %26, %20 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32> loc(#loc22)
    %28 = tt.load %27, %25, %cst_1 : tensor<16x128x!tt.ptr<bf16>> loc(#loc23)
    %29 = arith.extf %28 : tensor<16x128xbf16> to tensor<16x128xf32> loc(#loc24)
    %30 = tt.splat %arg6 : f32 -> tensor<16x128xf32> loc(#loc24)
    %31 = arith.mulf %29, %30 : tensor<16x128xf32> loc(#loc24)
    %32 = arith.truncf %31 : tensor<16x128xf32> to tensor<16x128xbf16> loc(#loc25)
    %33 = arith.muli %1, %arg14 : i32 loc(#loc26)
    %34 = tt.splat %arg15 : i32 -> tensor<16x1xi32> loc(#loc27)
    %35 = arith.muli %10, %34 : tensor<16x1xi32> loc(#loc27)
    %36 = tt.splat %33 : i32 -> tensor<16x1xi32> loc(#loc28)
    %37 = arith.addi %36, %35 : tensor<16x1xi32> loc(#loc28)
    %38 = tt.broadcast %37 : tensor<16x1xi32> -> tensor<16x128xi32> loc(#loc29)
    %39 = arith.addi %38, %19 : tensor<16x128xi32> loc(#loc29)
    %40 = arith.muli %0, %arg16 : i32 loc(#loc30)
    %41 = tt.addptr %arg4, %40 : !tt.ptr<i32>, i32 loc(#loc31)
    %42 = arith.extsi %6 : tensor<16xi32> to tensor<16xi64> loc(#loc32)
    %43 = tt.splat %3 : i64 -> tensor<16x1xi64> loc(#loc33)
    %44 = arith.cmpi slt, %10, %cst_6 : tensor<16x1xi32> loc(#loc34)
    %45 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>> loc(#loc35)
    %46 = tt.splat %3 : i64 -> tensor<1x16xi64> loc(#loc36)
    %47 = tt.broadcast %21 : tensor<16x1xi1> -> tensor<16x16xi1> loc(#loc37)
    %48 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>> loc(#loc38)
    %49:3 = scf.for %arg17 = %c0_i64 to %5 step %c1_i64 iter_args(%arg18 = %cst_7, %arg19 = %cst_0, %arg20 = %cst) -> (tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32>)  : i64 {
      %63 = tt.addptr %41, %arg17 : !tt.ptr<i32>, i64 loc(#loc40)
      %64 = tt.load %63 : !tt.ptr<i32> loc(#loc41)
      %65 = arith.muli %64, %arg13 : i32 loc(#loc42)
      %66 = tt.splat %65 : i32 -> tensor<16x128xi32> loc(#loc43)
      %67 = arith.addi %66, %39 : tensor<16x128xi32> loc(#loc43)
      %68 = arith.muli %arg17, %c16_i64 : i64 loc(#loc44)
      %69 = tt.splat %68 : i64 -> tensor<16xi64> loc(#loc32)
      %70 = arith.addi %69, %42 : tensor<16xi64> loc(#loc32)
      %71 = tt.expand_dims %70 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64> loc(#loc45)
      %72 = arith.cmpi slt, %71, %43 : tensor<16x1xi64> loc(#loc33)
      %73 = arith.andi %72, %44 : tensor<16x1xi1> loc(#loc46)
      %74 = tt.broadcast %73 : tensor<16x1xi1> -> tensor<16x128xi1> loc(#loc47)
      %75 = arith.andi %74, %24 : tensor<16x128xi1> loc(#loc47)
      %76 = tt.addptr %45, %67 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32> loc(#loc35)
      %77 = tt.load %76, %75, %cst_1 : tensor<16x128x!tt.ptr<bf16>> loc(#loc48)
      %78 = tt.trans %77 {order = array<i32: 1, 0>} : tensor<16x128xbf16> -> tensor<128x16xbf16> loc(#loc49)
      %79 = tt.dot %32, %78, %cst_5 : tensor<16x128xbf16> * tensor<128x16xbf16> -> tensor<16x16xf32> loc(#loc49)
      %80 = tt.expand_dims %70 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64> loc(#loc50)
      %81 = arith.cmpi slt, %80, %46 : tensor<1x16xi64> loc(#loc36)
      %82 = tt.broadcast %81 : tensor<1x16xi1> -> tensor<16x16xi1> loc(#loc37)
      %83 = arith.andi %47, %82 : tensor<16x16xi1> loc(#loc37)
      %84 = arith.select %83, %79, %cst_4 : tensor<16x16xi1>, tensor<16x16xf32> loc(#loc51)
      %85 = arith.select %83, %84, %cst_4 : tensor<16x16xi1>, tensor<16x16xf32> loc(#loc52)
      %86 = "tt.reduce"(%85) <{axis = 1 : i32}> ({
      ^bb0(%arg21: f32 loc(callsite(#loc1 at #loc54)), %arg22: f32 loc(callsite(#loc1 at #loc54))):
        %107 = arith.maxnumf %arg21, %arg22 : f32 loc(#loc95)
        tt.reduce.return %107 : f32 loc(#loc88)
      }) : (tensor<16x16xf32>) -> tensor<16xf32> loc(#loc88)
      %87 = arith.maxnumf %86, %arg20 : tensor<16xf32> loc(#loc56)
      %88 = tt.expand_dims %87 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32> loc(#loc57)
      %89 = tt.broadcast %88 : tensor<16x1xf32> -> tensor<16x16xf32> loc(#loc58)
      %90 = arith.subf %85, %89 : tensor<16x16xf32> loc(#loc58)
      %91 = arith.mulf %90, %cst_3 : tensor<16x16xf32> loc(#loc59)
      %92 = math.exp2 %91 : tensor<16x16xf32> loc(#loc60)
      %93 = arith.subf %arg20, %87 : tensor<16xf32> loc(#loc61)
      %94 = arith.mulf %93, %cst_2 : tensor<16xf32> loc(#loc62)
      %95 = math.exp2 %94 : tensor<16xf32> loc(#loc63)
      %96 = tt.expand_dims %95 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32> loc(#loc64)
      %97 = tt.broadcast %96 : tensor<16x1xf32> -> tensor<16x128xf32> loc(#loc65)
      %98 = arith.mulf %arg18, %97 : tensor<16x128xf32> loc(#loc65)
      %99 = tt.addptr %48, %67 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32> loc(#loc38)
      %100 = tt.load %99, %75, %cst_1 : tensor<16x128x!tt.ptr<bf16>> loc(#loc66)
      %101 = arith.truncf %92 : tensor<16x16xf32> to tensor<16x16xbf16> loc(#loc67)
      %102 = tt.dot %101, %100, %98 : tensor<16x16xbf16> * tensor<16x128xbf16> -> tensor<16x128xf32> loc(#loc68)
      %103 = arith.mulf %arg19, %95 : tensor<16xf32> loc(#loc69)
      %104 = arith.extf %101 : tensor<16x16xbf16> to tensor<16x16xf32> loc(#loc91)
      %105 = "tt.reduce"(%104) <{axis = 1 : i32}> ({
      ^bb0(%arg21: f32 loc(callsite(#loc1 at #loc71)), %arg22: f32 loc(callsite(#loc1 at #loc71))):
        %107 = arith.addf %arg21, %arg22 : f32 loc(#loc96)
        tt.reduce.return %107 : f32 loc(#loc92)
      }) : (tensor<16x16xf32>) -> tensor<16xf32> loc(#loc92)
      %106 = arith.addf %103, %105 : tensor<16xf32> loc(#loc74)
      scf.yield %102, %106, %87 : tensor<16x128xf32>, tensor<16xf32>, tensor<16xf32> loc(#loc75)
    } loc(#loc39)
    %50 = tt.expand_dims %49#1 {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32> loc(#loc76)
    %51 = tt.broadcast %50 : tensor<16x1xf32> -> tensor<16x128xf32> loc(#loc77)
    %52 = arith.divf %49#0, %51 : tensor<16x128xf32> loc(#loc77)
    %53 = arith.muli %0, %arg9 : i32 loc(#loc78)
    %54 = tt.splat %arg10 : i32 -> tensor<16x1xi32> loc(#loc79)
    %55 = arith.muli %12, %54 : tensor<16x1xi32> loc(#loc79)
    %56 = tt.splat %53 : i32 -> tensor<16x1xi32> loc(#loc80)
    %57 = arith.addi %56, %55 : tensor<16x1xi32> loc(#loc80)
    %58 = tt.broadcast %57 : tensor<16x1xi32> -> tensor<16x128xi32> loc(#loc81)
    %59 = arith.addi %58, %19 : tensor<16x128xi32> loc(#loc81)
    %60 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>> loc(#loc82)
    %61 = tt.addptr %60, %59 : tensor<16x128x!tt.ptr<bf16>>, tensor<16x128xi32> loc(#loc82)
    %62 = arith.truncf %52 : tensor<16x128xf32> to tensor<16x128xbf16> loc(#loc83)
    tt.store %61, %62, %25 : tensor<16x128x!tt.ptr<bf16>> loc(#loc84)
    tt.return loc(#loc85)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/workspace/pa_decode_rework.py":192:28)
#loc3 = loc("/workspace/pa_decode_rework.py":193:32)
#loc4 = loc("/workspace/pa_decode_rework.py":197:37)
#loc5 = loc("/workspace/pa_decode_rework.py":197:22)
#loc6 = loc("/usr/local/lib/python3.12/dist-packages/triton/language/standard.py":40:22)
#loc7 = loc("/workspace/pa_decode_rework.py":199:35)
#loc8 = loc("/usr/local/lib/python3.12/dist-packages/triton/language/standard.py":40:28)
#loc9 = loc("/workspace/pa_decode_rework.py":201:28)
#loc10 = loc("/workspace/pa_decode_rework.py":202:32)
#loc11 = loc("/workspace/pa_decode_rework.py":216:18)
#loc12 = loc("/workspace/pa_decode_rework.py":217:25)
#loc13 = loc("/workspace/pa_decode_rework.py":217:51)
#loc14 = loc("/workspace/pa_decode_rework.py":217:40)
#loc15 = loc("/workspace/pa_decode_rework.py":217:63)
#loc16 = loc("/workspace/pa_decode_rework.py":217:10)
#loc17 = loc("/workspace/pa_decode_rework.py":218:23)
#loc18 = loc("/workspace/pa_decode_rework.py":218:10)
#loc19 = loc("/workspace/pa_decode_rework.py":222:36)
#loc20 = loc("/workspace/pa_decode_rework.py":222:77)
#loc21 = loc("/workspace/pa_decode_rework.py":222:53)
#loc22 = loc("/workspace/pa_decode_rework.py":224:24)
#loc23 = loc("/workspace/pa_decode_rework.py":224:16)
#loc24 = loc("/workspace/pa_decode_rework.py":225:13)
#loc25 = loc("/workspace/pa_decode_rework.py":225:23)
#loc26 = loc("/workspace/pa_decode_rework.py":232:22)
#loc27 = loc("/workspace/pa_decode_rework.py":233:30)
#loc28 = loc("/workspace/pa_decode_rework.py":233:10)
#loc29 = loc("/workspace/pa_decode_rework.py":234:10)
#loc30 = loc("/workspace/pa_decode_rework.py":236:51)
#loc31 = loc("/workspace/pa_decode_rework.py":236:41)
#loc32 = loc("/workspace/pa_decode_rework.py":241:39)
#loc33 = loc("/workspace/pa_decode_rework.py":243:37)
#loc34 = loc("/workspace/pa_decode_rework.py":244:35)
#loc35 = loc("/workspace/pa_decode_rework.py":249:36)
#loc36 = loc("/workspace/pa_decode_rework.py":256:76)
#loc37 = loc("/workspace/pa_decode_rework.py":256:52)
#loc38 = loc("/workspace/pa_decode_rework.py":279:36)
#loc39 = loc("/workspace/pa_decode_rework.py":238:19)
#loc40 = loc("/workspace/pa_decode_rework.py":239:50)
#loc41 = loc("/workspace/pa_decode_rework.py":239:30)
#loc42 = loc("/workspace/pa_decode_rework.py":240:36)
#loc43 = loc("/workspace/pa_decode_rework.py":240:49)
#loc44 = loc("/workspace/pa_decode_rework.py":241:27)
#loc45 = loc("/workspace/pa_decode_rework.py":243:26)
#loc46 = loc("/workspace/pa_decode_rework.py":244:15)
#loc47 = loc("/workspace/pa_decode_rework.py":245:15)
#loc48 = loc("/workspace/pa_decode_rework.py":249:22)
#loc49 = loc("/workspace/pa_decode_rework.py":254:23)
#loc50 = loc("/workspace/pa_decode_rework.py":256:65)
#loc51 = loc("/workspace/pa_decode_rework.py":258:12)
#loc52 = loc("/workspace/pa_decode_rework.py":269:12)
#loc53 = loc("/usr/local/lib/python3.12/dist-packages/triton/language/standard.py":184:40)
#loc55 = loc("/usr/local/lib/python3.12/dist-packages/triton/language/standard.py":163:27)
#loc56 = loc("/workspace/pa_decode_rework.py":271:55)
#loc57 = loc("/workspace/pa_decode_rework.py":274:45)
#loc58 = loc("/workspace/pa_decode_rework.py":274:31)
#loc59 = loc("/workspace/pa_decode_rework.py":274:57)
#loc60 = loc("/workspace/pa_decode_rework.py":274:25)
#loc61 = loc("/workspace/pa_decode_rework.py":275:42)
#loc62 = loc("/workspace/pa_decode_rework.py":275:59)
#loc63 = loc("/workspace/pa_decode_rework.py":275:29)
#loc64 = loc("/workspace/pa_decode_rework.py":276:21)
#loc65 = loc("/workspace/pa_decode_rework.py":276:15)
#loc66 = loc("/workspace/pa_decode_rework.py":279:22)
#loc67 = loc("/workspace/pa_decode_rework.py":283:17)
#loc68 = loc("/workspace/pa_decode_rework.py":284:25)
#loc69 = loc("/workspace/pa_decode_rework.py":286:28)
#loc70 = loc("/usr/local/lib/python3.12/dist-packages/triton/language/standard.py":266:46)
#loc72 = loc("/usr/local/lib/python3.12/dist-packages/triton/language/standard.py":267:36)
#loc73 = loc("/usr/local/lib/python3.12/dist-packages/triton/language/standard.py":256:15)
#loc74 = loc("/workspace/pa_decode_rework.py":286:36)
#loc75 = loc("/workspace/pa_decode_rework.py":287:8)
#loc76 = loc("/workspace/pa_decode_rework.py":289:24)
#loc77 = loc("/workspace/pa_decode_rework.py":289:16)
#loc78 = loc("/workspace/pa_decode_rework.py":292:18)
#loc79 = loc("/workspace/pa_decode_rework.py":293:63)
#loc80 = loc("/workspace/pa_decode_rework.py":293:10)
#loc81 = loc("/workspace/pa_decode_rework.py":294:10)
#loc82 = loc("/workspace/pa_decode_rework.py":298:23)
#loc83 = loc("/workspace/pa_decode_rework.py":298:40)
#loc84 = loc("/workspace/pa_decode_rework.py":298:33)
#loc85 = loc("/workspace/pa_decode_rework.py":298:4)
#loc86 = loc(callsite(#loc6 at #loc7))
#loc87 = loc(callsite(#loc8 at #loc7))
#loc88 = loc(callsite(#loc53 at #loc54))
#loc90 = loc(callsite(#loc55 at #loc53))
#loc91 = loc(callsite(#loc70 at #loc71))
#loc92 = loc(callsite(#loc72 at #loc71))
#loc94 = loc(callsite(#loc73 at #loc72))
#loc95 = loc(callsite(#loc90 at #loc54))
#loc96 = loc(callsite(#loc94 at #loc71))
