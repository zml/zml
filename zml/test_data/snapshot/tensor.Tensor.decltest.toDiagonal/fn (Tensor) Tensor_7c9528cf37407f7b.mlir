module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<2x2xui8> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<2x2x2xui8> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<ui8>
    %0 = stablehlo.iota dim = 1 : tensor<2x2x2xi32>
    %1 = stablehlo.iota dim = 2 : tensor<2x2x2xi32>
    %2 = stablehlo.compare  EQ, %0, %1,  SIGNED : (tensor<2x2x2xi32>, tensor<2x2x2xi32>) -> tensor<2x2x2xi1>
    %3 = stablehlo.reshape %arg0 : (tensor<2x2xui8>) -> tensor<2x2x1xui8>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<2x2x1xui8>) -> tensor<2x2x2xui8>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui8>) -> tensor<2x2x2xui8>
    %6 = stablehlo.select %2, %4, %5 : tensor<2x2x2xi1>, tensor<2x2x2xui8>
    return %6 : tensor<2x2x2xui8>
  }
}
