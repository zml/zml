module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x5x4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<5x4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<[1.000000e+00, 0.00999999977]> : tensor<2xf32>
    %0 = stablehlo.slice %arg0 [0:1, 0:5, 0:2] : (tensor<1x5x4xf32>) -> tensor<1x5x2xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf32>
    %2 = stablehlo.slice %arg0 [0:1, 0:5, 2:4] : (tensor<1x5x4xf32>) -> tensor<1x5x2xf32>
    %3 = stablehlo.reshape %2 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf32>
    %4 = stablehlo.concatenate %1, %3, dim = 2 : (tensor<1x5x2xf32>, tensor<1x5x2xf32>) -> tensor<1x5x4xf32>
    %5 = stablehlo.slice %4 [0:1, 0:5, 0:2] : (tensor<1x5x4xf32>) -> tensor<1x5x2xf32>
    %6 = stablehlo.reshape %5 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf32>
    %7 = stablehlo.iota dim = 0 : tensor<5xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<5xf32>) -> tensor<5x2xf32>
    %9 = stablehlo.broadcast_in_dim %cst, dims = [1] : (tensor<2xf32>) -> tensor<5x2xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<5x2xf32>
    %11 = stablehlo.cosine %10 : tensor<5x2xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1, 2] : (tensor<5x2xf32>) -> tensor<1x5x2xf32>
    %13 = stablehlo.multiply %6, %12 : tensor<1x5x2xf32>
    %14 = stablehlo.slice %4 [0:1, 0:5, 2:4] : (tensor<1x5x4xf32>) -> tensor<1x5x2xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf32>
    %16 = stablehlo.sine %10 : tensor<5x2xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [1, 2] : (tensor<5x2xf32>) -> tensor<1x5x2xf32>
    %18 = stablehlo.multiply %15, %17 : tensor<1x5x2xf32>
    %19 = stablehlo.subtract %13, %18 : tensor<1x5x2xf32>
    %20 = stablehlo.multiply %6, %17 : tensor<1x5x2xf32>
    %21 = stablehlo.multiply %15, %12 : tensor<1x5x2xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x5x2xf32>
    %23 = stablehlo.concatenate %19, %22, dim = 2 : (tensor<1x5x2xf32>, tensor<1x5x2xf32>) -> tensor<1x5x4xf32>
    %24 = stablehlo.reshape %23 : (tensor<1x5x4xf32>) -> tensor<5x4xf32>
    %25 = stablehlo.slice %24 [0:5, 0:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %26 = stablehlo.reshape %25 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %27 = stablehlo.slice %24 [0:5, 2:4] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %28 = stablehlo.reshape %27 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %29 = stablehlo.concatenate %26, %28, dim = 1 : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x4xf32>
    return %29 : tensor<5x4xf32>
  }
}
