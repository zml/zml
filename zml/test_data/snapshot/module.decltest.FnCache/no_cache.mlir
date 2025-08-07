module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<2x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg4: tensor<3x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg5: tensor<3xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg6: tensor<2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<3xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.dot_general %arg0, %arg6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %1 = stablehlo.add %0, %arg1 : tensor<2xf16>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<2xf16>
    %3 = stablehlo.maximum %1, %2 : tensor<2xf16>
    %4 = stablehlo.dot_general %arg2, %3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %5 = stablehlo.add %4, %arg3 : tensor<2xf16>
    %6 = stablehlo.maximum %5, %2 : tensor<2xf16>
    %7 = stablehlo.dot_general %arg4, %6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x2xf16>, tensor<2xf16>) -> tensor<3xf16>
    %8 = stablehlo.add %7, %arg5 : tensor<3xf16>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<3xf16>
    %10 = stablehlo.maximum %8, %9 : tensor<3xf16>
    return %10 : tensor<3xf16>
  }
}
