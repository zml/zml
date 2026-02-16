
python examples/python/flux2-pipeline/flux2-pipeline.py

rm *.npy &&
bazel run //examples/flux -- --model=/Users/kevin/FLUX.2-klein-4B &&
python verify_npy.py /Users/kevin/zml/flux_klein_zml_result.npy
