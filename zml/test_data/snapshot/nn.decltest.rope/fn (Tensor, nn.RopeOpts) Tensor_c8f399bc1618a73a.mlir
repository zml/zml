module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<1x5x4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<5x4xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %cst = stablehlo.constant dense<[1.000000e+00, 0.00999999977]> : tensor<2xf32>
    %0 = stablehlo.slice %arg0 [0:1, 0:5, 0:2] : (tensor<1x5x4xf32>) -> tensor<1x5x2xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x5x2xf32>) -> tensor<1x5x2x1xf32>
    %3 = stablehlo.slice %arg0 [0:1, 0:5, 2:4] : (tensor<1x5x4xf32>) -> tensor<1x5x2xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf32>
    %5 = stablehlo.reshape %4 : (tensor<1x5x2xf32>) -> tensor<1x5x2x1xf32>
    %6 = stablehlo.concatenate %2, %5, dim = 3 : (tensor<1x5x2x1xf32>, tensor<1x5x2x1xf32>) -> tensor<1x5x2x2xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x5x2x2xf32>) -> tensor<1x5x4xf32>
    %8 = stablehlo.slice %7 [0:1, 0:5, 0:4:2] : (tensor<1x5x4xf32>) -> tensor<1x5x2xf32>
    %9 = stablehlo.reshape %8 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf32>
    %10 = stablehlo.iota dim = 0 : tensor<5xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<5xf32>) -> tensor<5x2xf32>
    %12 = stablehlo.broadcast_in_dim %cst, dims = [1] : (tensor<2xf32>) -> tensor<5x2xf32>
    %13 = stablehlo.multiply %11, %12 : tensor<5x2xf32>
    %14 = stablehlo.cosine %13 : tensor<5x2xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<5x2xf32>) -> tensor<1x5x2xf32>
    %16 = stablehlo.multiply %9, %15 : tensor<1x5x2xf32>
    %17 = stablehlo.slice %7 [0:1, 0:5, 1:4:2] : (tensor<1x5x4xf32>) -> tensor<1x5x2xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x5x2xf32>) -> tensor<1x5x2xf32>
    %19 = stablehlo.sine %13 : tensor<5x2xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [1, 2] : (tensor<5x2xf32>) -> tensor<1x5x2xf32>
    %21 = stablehlo.multiply %18, %20 : tensor<1x5x2xf32>
    %22 = stablehlo.subtract %16, %21 : tensor<1x5x2xf32>
    %23 = stablehlo.reshape %22 : (tensor<1x5x2xf32>) -> tensor<1x5x2x1xf32>
    %24 = stablehlo.multiply %9, %20 : tensor<1x5x2xf32>
    %25 = stablehlo.multiply %18, %15 : tensor<1x5x2xf32>
    %26 = stablehlo.add %24, %25 : tensor<1x5x2xf32>
    %27 = stablehlo.reshape %26 : (tensor<1x5x2xf32>) -> tensor<1x5x2x1xf32>
    %28 = stablehlo.concatenate %23, %27, dim = 3 : (tensor<1x5x2x1xf32>, tensor<1x5x2x1xf32>) -> tensor<1x5x2x2xf32>
    %29 = stablehlo.reshape %28 : (tensor<1x5x2x2xf32>) -> tensor<1x5x4xf32>
    %30 = stablehlo.reshape %29 : (tensor<1x5x4xf32>) -> tensor<5x4xf32>
    %31 = stablehlo.slice %30 [0:5, 0:4:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %32 = stablehlo.reshape %31 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %33 = stablehlo.slice %30 [0:5, 1:4:2] : (tensor<5x4xf32>) -> tensor<5x2xf32>
    %34 = stablehlo.reshape %33 : (tensor<5x2xf32>) -> tensor<5x2xf32>
    %35 = stablehlo.concatenate %32, %34, dim = 1 : (tensor<5x2xf32>, tensor<5x2xf32>) -> tensor<5x4xf32>
    return %35 : tensor<5x4xf32>
  }
}
