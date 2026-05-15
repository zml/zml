# DFlash Model Notes

The repository checkout on the `9960x-5090x2` remote is at:

```text
~/zml
```

Use these model paths on that remote:

```text
DFlash model: /var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/
Target model: /var/models/meta-llama/Llama-3.1-8B-Instruct/
```

Typical CUDA run command:

```bash
CUDA_VISIBLE_DEVICES=1 bazel run --@zml//platforms:cuda=true //examples/dflash:dflash -- --model=/var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/ --target-model=/var/models/meta-llama/Llama-3.1-8B-Instruct/ --prompt="Give me a detailed account of the history of the Richelieu-Drouot part of Paris." --max-seq-len=4096
```
