module @zml attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<i64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<i64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg2: tensor<i64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg3: tensor<i64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<i64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, tensor<i64> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) {
    %c = stablehlo.constant dense<2> : tensor<i64>
    %0 = stablehlo.multiply %arg2, %c : tensor<i64>
    %1:2 = stablehlo.while(%iterArg = %0, %iterArg_0 = %arg3) : tensor<i64>, tensor<i64> attributes {operandSegmentSizes = dense<2> : tensor<1xi32>}
    cond {
      %2 = stablehlo.compare  LT, %iterArg, %arg1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %2 {operandSegmentSizes = dense<1> : tensor<1xi32>} : tensor<i1>
    } do {
      %2 = stablehlo.add %iterArg, %arg0 : tensor<i64>
      %3 = stablehlo.add %iterArg_0, %iterArg : tensor<i64>
      stablehlo.return %2, %3 {operandSegmentSizes = dense<2> : tensor<1xi32>} : tensor<i64>, tensor<i64>
    }
    return %1#0, %1#1 : tensor<i64>, tensor<i64>
  }
}
