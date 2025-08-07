module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<2x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<3x2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<2xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<3xf16> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %0 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f16>) -> tensor<2xf16>
    %2 = stablehlo.add %0, %1 : tensor<2xf16>
    %3 = stablehlo.dot_general %arg1, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf16>, tensor<2xf16>) -> tensor<2xf16>
    %4 = stablehlo.add %3, %1 : tensor<2xf16>
    %5 = stablehlo.dot_general %arg2, %4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x2xf16>, tensor<2xf16>) -> tensor<3xf16>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f16>) -> tensor<3xf16>
    %7 = stablehlo.add %5, %6 : tensor<3xf16>
    return %7 : tensor<3xf16>
  }
}
