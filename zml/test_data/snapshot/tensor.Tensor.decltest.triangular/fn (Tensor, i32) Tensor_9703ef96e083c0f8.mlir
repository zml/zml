module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<3x3xui8> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<3x3xui8> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<0> : tensor<ui8>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.iota dim = 0 : tensor<3x3xi32>
    %1 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
    %2 = stablehlo.add %0, %1 : tensor<3x3xi32>
    %3 = stablehlo.iota dim = 1 : tensor<3x3xi32>
    %4 = stablehlo.compare  GE, %2, %3,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %5 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui8>) -> tensor<3x3xui8>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<3x3xi1>, tensor<3x3xui8>
    return %6 : tensor<3x3xui8>
  }
}
