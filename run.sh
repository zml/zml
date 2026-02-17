export CUDA_VISIBLE_DEVICES=0

#  -c opt 

# hf download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B
# /Users/kevin/zml
# --output-image-path=/home/kevin/flux_klein_zml_result.png \
# --output-image-path=/Users/kevin/zml/flux_klein_zml_result.png \

# --config=release
# --run_under="lldb --"
# --run_under="lldb --" \

# --run_under="lldb --batch -o run -k bt -k exit --" \

# --kitty-output \
# HLD

# bazel run //examples/flux \
#     --//platforms:cuda=false -- \
#     --model=/Users/kevin/FLUX.2-klein-4B \
#     --prompt="A photo of a cat on a bed" \
#     --resolution=SD \
#     --num-inference-steps=4 \
#     --random-seed=0 \
#     --output-image-path=/Users/kevin/zml/flux_klein_zml_result.png \
#     --generator-type=torch

# bazel test //examples/flux:test_scheduler_consistency

# --output-image-path=/home/kevin/flux_klein_zml_result.png \

bazel run //examples/flux \
    --config=release \
    --//platforms:cuda=true -- \
    --model=/var/models/black-forest-labs/FLUX.2-klein-4B/ \
    --prompt="A photo of a cat on a bed" \
    --output-image-path=/home/kevin/flux_klein_zml_result.png \
    --resolution=SD \
    --num-inference-steps=4 \
    --random-seed=0 \
    --generator-type=torch

# --seqlen=256 \

# bazel run //examples/flux --//platforms:cuda=true -- \
#     --model=/var/models/black-forest-labs/FLUX.2-klein-4B/ \
#     --prompt="A photo of a cat" \
#     --output-image-path=/home/kevin/flux_klein_zml_result.png \
#     --resolution=HD \
#     --num-inference-steps=4 \
#     --random-seed=0 \
#     --generator-type=accelerator_box_muller

# bazel run //examples/flux --//platforms:cuda=true -- \
#     --model=/var/models/black-forest-labs/FLUX.2-klein-4B/ \
#     --prompt="A photo of a cat" \
#     --output-image-path=/home/kevin/flux_klein_zml_result.png \
#     --resolution=HD \
#     --num-inference-steps=4 \
#     --random-seed=0 \
#     --generator-type=accelerator_marsaglia
