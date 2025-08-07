module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2xui64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}", tf.aliasing_output = 1 : i32}) -> (tensor<i32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<2xui64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<-1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0.99999988> : tensor<f32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %c_4 = stablehlo.constant dense<9> : tensor<ui32>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %0 = stablehlo.iota dim = 0 : tensor<4xi32>
    %1:2 = "stablehlo.sort"(%arg0, %0) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
      %33 = stablehlo.compare  GT, %arg2, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %33 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    }) {operandSegmentSizes = dense<2> : tensor<1xi32>} : (tensor<4xf32>, tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    %2 = stablehlo.slice %1#1 [0:4] : (tensor<4xi32>) -> tensor<4xi32>
    %3 = stablehlo.reshape %2 : (tensor<4xi32>) -> tensor<4xi32>
    %4 = stablehlo.slice %1#0 [0:4] : (tensor<4xf32>) -> tensor<4xf32>
    %5 = stablehlo.reshape %4 : (tensor<4xf32>) -> tensor<4xf32>
    %6 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<4xf32>
    %output_state, %output = stablehlo.rng_bit_generator %arg1, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %8 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %9 = stablehlo.shift_right_logical %output, %8 : tensor<4xui32>
    %10 = stablehlo.bitcast_convert %cst_3 : (tensor<f32>) -> tensor<ui32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<ui32>) -> tensor<4xui32>
    %12 = stablehlo.or %9, %11 : tensor<4xui32>
    %13 = stablehlo.bitcast_convert %12 : (tensor<4xui32>) -> tensor<4xf32>
    %14 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %15 = stablehlo.subtract %13, %14 : tensor<4xf32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<4xf32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %19 = stablehlo.add %17, %18 : tensor<4xf32>
    %20 = stablehlo.log %19 : tensor<4xf32>
    %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<4xf32>
    %23 = stablehlo.log %22 : tensor<4xf32>
    %24 = stablehlo.multiply %23, %21 : tensor<4xf32>
    %25 = stablehlo.add %7, %24 : tensor<4xf32>
    %26:2 = stablehlo.reduce(%25 init: %cst), (%0 init: %c) across dimensions = [0] {operandSegmentSizes = dense<2> : tensor<2xi32>} : (tensor<4xf32>, tensor<4xi32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
     reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %33 = stablehlo.compare  GT, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %34 = stablehlo.compare  NE, %arg2, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %35 = stablehlo.or %33, %34 : tensor<i1>
      %36 = stablehlo.select %35, %arg2, %arg4 : tensor<i1>, tensor<f32>
      %37 = stablehlo.compare  EQ, %arg2, %arg4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %38 = stablehlo.compare  LT, %arg3, %arg5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %39 = stablehlo.and %37, %38 : tensor<i1>
      %40 = stablehlo.or %35, %39 : tensor<i1>
      %41 = stablehlo.select %40, %arg3, %arg5 : tensor<i1>, tensor<i32>
      stablehlo.return %36, %41 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<f32>, tensor<i32>
    }
    %27 = stablehlo.broadcast_in_dim %26#1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %28 = stablehlo.reshape %27 : (tensor<1xi32>) -> tensor<i32>
    %29 = stablehlo.reshape %28 : (tensor<i32>) -> tensor<1xi32>
    %30 = stablehlo.reshape %29 : (tensor<1xi32>) -> tensor<i32>
    %31 = stablehlo.dynamic_slice %3, %30, sizes = [1] {operandSegmentSizes = dense<1> : tensor<2xi32>} : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
    %32 = stablehlo.reshape %31 : (tensor<1xi32>) -> tensor<i32>
    return %32, %output_state : tensor<i32>, tensor<2xui64>
  }
}
