CUDA_VISIBLE_DEVICES=0 bazel run --config=remote --config=debug --@zml//platforms:cuda=true --@zml//platforms:cpu=false //zml:test

