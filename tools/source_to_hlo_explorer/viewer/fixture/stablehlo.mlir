#loc = [unknown]
module @zml_source_to_hlo_poc attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @replicated = <["bus"=4]> [unknown]
  func.func public @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@replicated, [{}], replicated={"bus"}>} [unknown], %arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@replicated, [{}], replicated={"bus"}>} [unknown]) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@replicated, [{}], replicated={"bus"}>}) {
    %cst = stablehlo.constant {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.1"}} dense<2.000000e+00> : tensor<f32> "zml.stable_op.1"(#loc1)
    %0 = stablehlo.add %arg0, %arg1 {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.0"}} : tensor<4xf32> "zml.stable_op.0"(#loc2)
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.2"}} : (tensor<f32>) -> tensor<4xf32> "zml.stable_op.2"(#loc1)
    %2 = stablehlo.multiply %0, %1 {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.3"}} : tensor<4xf32> "zml.stable_op.3"(#loc1)
    return %2 : tensor<4xf32> [unknown]
  } [unknown]
} [unknown]
#loc1 = source.zig:5:21
#loc2 = source.zig:4:17
#loc3 = "zml.stable_op.1"(#loc1)
#loc4 = "zml.stable_op.0"(#loc2)
#loc5 = "zml.stable_op.2"(#loc1)
#loc6 = "zml.stable_op.3"(#loc1)
