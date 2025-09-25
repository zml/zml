# GptOss

Download the model (you can change 20b to 120b for the bigger version if you have the required hardware).

```
bazel run //tools/hf -- download openai/gpt-oss-20b --local-dir /tmp/models/openai/gpt-oss-20b --exclude metal/ original/
```

Run the model, use `--@zml//runtimes:xxx` flags to target the relevant platform for your hardware.

```
bazel run --config=release //examples/gpt_oss \
	--@zml//runtimes:cuda=true \
	--@zml//runtimes:cpu=false \
	--@zml//runtimes:tpu=false \
	--@zml//runtimes:rocm=false \
	-- --hf-model-path=/tmp/models/openai/gpt-oss-20b/ \
	--prompt='What are some spooky funfacts about animal for a Halloween party ?'
```
