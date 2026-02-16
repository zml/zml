export CUDA_VISIBLE_DEVICES=1

#  -c opt 

# hf download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B
# /Users/kevin/zml

bazel run //examples/flux --//platforms:cuda=false -- \
    --model=/Users/kevin/zml \
    --prompt="A photo of a cat" \
    --output-image-path=/home/kevin/flux_klein_zml_result.png \
    --kitty-output \
    --resolution=HLD \
    --num-inference-steps=1 \
    --random-seed=0 \
    --generator-type=torch

# bazel run //examples/flux --//platforms:cuda=false -- \
#     --model=/var/models/black-forest-labs/FLUX.2-klein-4B/ \
#     --prompt="A photo of a cat" \
#     --output-image-path=/home/kevin/flux_klein_zml_result.png \
#     --kitty-output \
#     --resolution=SD \
#     --num-inference-steps=4 \
#     --random-seed=0 \
#     --generator-type=torch

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
