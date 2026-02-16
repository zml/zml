export CUDA_VISIBLE_DEVICES=1

#  -c opt 

bazel run //examples/flux --//platforms:cuda=true -- \
    --model=/var/models/black-forest-labs/FLUX.2-klein-4B/ \
    --prompt="A photo of a cat" \
    --output-image-path=/home/kevin/flux_klein_zml_result.png \
    --kitty-output \
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