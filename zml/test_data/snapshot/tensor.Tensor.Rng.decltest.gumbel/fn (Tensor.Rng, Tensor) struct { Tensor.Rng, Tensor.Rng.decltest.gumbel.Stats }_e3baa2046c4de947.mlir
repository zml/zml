module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2xui64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i32}, %arg1: tensor<4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2xui64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<1xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<1.024000e+03> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<ui64>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_2 = stablehlo.constant dense<[1, 65536, 4294967296, 281474976710656]> : tensor<4xui64>
    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %cst_4 = stablehlo.constant dense<4.096000e+03> : tensor<f32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_6 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %cst_7 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %cst_8 = stablehlo.constant dense<0.99999988> : tensor<f32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c_10 = stablehlo.constant dense<9> : tensor<ui32>
    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<1024x4xui32>)
    %0 = stablehlo.broadcast_in_dim %c_10, dims = [] : (tensor<ui32>) -> tensor<1024x4xui32>
    %1 = stablehlo.shift_right_logical %output, %0 : tensor<1024x4xui32>
    %2 = stablehlo.bitcast_convert %cst_9 : (tensor<f32>) -> tensor<ui32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui32>) -> tensor<1024x4xui32>
    %4 = stablehlo.or %1, %3 : tensor<1024x4xui32>
    %5 = stablehlo.bitcast_convert %4 : (tensor<1024x4xui32>) -> tensor<1024x4xf32>
    %6 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1024x4xf32>
    %7 = stablehlo.subtract %5, %6 : tensor<1024x4xf32>
    %8 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1024x4xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1024x4xf32>
    %10 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1024x4xf32>
    %11 = stablehlo.add %9, %10 : tensor<1024x4xf32>
    %12 = stablehlo.log %11 : tensor<1024x4xf32>
    %13 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1024x4xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<1024x4xf32>
    %15 = stablehlo.log %14 : tensor<1024x4xf32>
    %16 = stablehlo.multiply %15, %13 : tensor<1024x4xf32>
    %17 = stablehlo.reshape %16 : (tensor<1024x4xf32>) -> tensor<4096xf32>
    %18 = stablehlo.reduce(%17 init: %cst_5) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<4096xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %45 = stablehlo.add %arg3, %arg2 : tensor<f32>
      stablehlo.return %45 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %21 = stablehlo.divide %19, %20 : tensor<1xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<1xf32>) -> tensor<4096xf32>
    %23 = stablehlo.subtract %17, %22 : tensor<4096xf32>
    %24 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<4096xf32>
    %25 = stablehlo.power %23, %24 : tensor<4096xf32>
    %26 = stablehlo.reduce(%25 init: %cst_5) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<4096xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %45 = stablehlo.add %arg3, %arg2 : tensor<f32>
      stablehlo.return %45 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<f32>
    }
    %27 = stablehlo.broadcast_in_dim %26, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %28 = stablehlo.divide %27, %20 : tensor<1xf32>
    %29 = stablehlo.log %arg1 : tensor<4xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [1] : (tensor<4xf32>) -> tensor<1024x4xf32>
    %31 = stablehlo.add %30, %16 : tensor<1024x4xf32>
    %32 = stablehlo.iota dim = 0 : tensor<4xi32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [1] : (tensor<4xi32>) -> tensor<1024x4xi32>
    %34:2 = stablehlo.reduce(%31 init: %cst_1), (%33 init: %c_0) across dimensions = [1] {operandSegmentSizes = dense<2> : tensor<2xi32>} : (tensor<1024x4xf32>, tensor<1024x4xi32>, tensor<f32>, tensor<i32>) -> (tensor<1024xf32>, tensor<1024xi32>)
     reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %45 = stablehlo.compare  GT, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %46 = stablehlo.compare  NE, %arg2, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %47 = stablehlo.or %45, %46 : tensor<i1>
      %48 = stablehlo.select %47, %arg2, %arg4 : tensor<i1>, tensor<f32>
      %49 = stablehlo.compare  EQ, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %50 = stablehlo.compare  LT, %arg3, %arg5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %51 = stablehlo.and %49, %50 : tensor<i1>
      %52 = stablehlo.or %47, %51 : tensor<i1>
      %53 = stablehlo.select %52, %arg3, %arg5 : tensor<i1>, tensor<i32>
      stablehlo.return %48, %53 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<f32>, tensor<i32>
    }
    %35 = stablehlo.broadcast_in_dim %34#1, dims = [0] : (tensor<1024xi32>) -> tensor<1024x1xi32>
    %36 = stablehlo.reshape %35 : (tensor<1024x1xi32>) -> tensor<1024xi32>
    %37 = "stablehlo.gather"(%c_2, %36) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xui64>, tensor<1024xi32>) -> tensor<1024xui64>
    %38 = stablehlo.reduce(%37 init: %c) across dimensions = [0] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<1024xui64>, tensor<ui64>) -> tensor<ui64>
     reducer(%arg2: tensor<ui64>, %arg3: tensor<ui64>)  {
      %45 = stablehlo.add %arg3, %arg2 : tensor<ui64>
      stablehlo.return %45 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<ui64>
    }
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<ui64>) -> tensor<1xui64>
    %40 = stablehlo.bitcast_convert %39 : (tensor<1xui64>) -> tensor<1x4xui16>
    %41 = stablehlo.reshape %40 : (tensor<1x4xui16>) -> tensor<4xui16>
    %42 = stablehlo.convert %41 : (tensor<4xui16>) -> tensor<4xf32>
    %43 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %44 = stablehlo.divide %42, %43 : tensor<4xf32>
    return %output_state, %21, %28, %44 : tensor<2xui64>, tensor<1xf32>, tensor<1xf32>, tensor<4xf32>
  }
}
