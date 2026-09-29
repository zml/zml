# Laya: typed decisions with ZML

[Laya](https://huggingface.co/convaiinnovations/laya) is an open-source (Apache-2.0) "System 1" decision model.
Instead of generating text, it answers **typed questions** about a state in a single encoder forward pass:

| type     | answer                                              |
|----------|-----------------------------------------------------|
| `choice` | one label from a fixed set, with probabilities      |
| `score`  | expected level on an ordinal scale                  |
| `noul`   | yes / no statement, `noul` = P(true)                |

Because nothing is generated, the output is always a valid answer: no parsing, no hallucinated labels.

The model is a ModernBERT-large encoder (28 layers, alternating global and sliding-window attention)
followed by a 2-layer transformer decision head that scores one `[MASK]` marker per option, plus an
act/escalate head. That is 421M parameters in total.

## Run

```bash
bazel run //examples/laya -- \
    --model=hf://convaiinnovations/laya \
    --state=@examples/laya/testdata/state.json \
    --questions=@examples/laya/testdata/questions.json
```

`--state` and `--questions` accept inline JSON or `@path`. `--state` also accepts plain text.

```json
{"model": "laya-rl-agent", "answers": {
  "team": {"type": "choice", "confidence": 0.93, "action": {"act_probability": 1.0000},
           "choice": "billing", "probabilities": {"billing": 0.9845, "engineering": 0.0045, "sales": 0.0058, "other": 0.0053}},
  "urgency": {"type": "score", "score": 1.7044, ...},
  "refund": {"type": "noul", "noul": 0.8871, ...}},
 "usage": {"input_tokens": 275, "output_tokens": 0}, ...}
```

The other checkpoints in the repository work the same way:
`--model=hf://convaiinnovations/laya/multilingual` and `--model=hf://convaiinnovations/laya/typed-decisions`.

Add the usual platform flags to run on an accelerator, e.g. `--@zml//platforms:cuda=true`.

### Options

| flag                   | default                   | meaning                                               |
|------------------------|---------------------------|-------------------------------------------------------|
| `--seqlen=<n>`         | `max_len` of the checkpoint | compiled sequence length; long states are truncated |
| `--dtype=f32\|f16\|bf16` | `f32`                   | activation dtype; use `f16`/`bf16` on GPUs only, the CPU backend emulates them slowly |
| `--show-prompt`        | off                       | log the token ids and marker positions                |

## Questions format

```json
{
  "team":    {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges, refunds", "sales": null}},
  "urgency": {"type": "score",  "instructions": "How urgent?", "criteria": ["can wait", "this week", "today"]},
  "refund":  {"type": "noul",   "instructions": "The customer asks to get money back."}
}
```

`choice` criteria can be a list of labels or an object mapping labels to descriptions.
`noul` criteria are optional: `{"false": "...", "true": "..."}`.

## Implementation notes

* `model.zig`: ModernBERT encoder and decision head. The global and sliding-window (±64 tokens) masks are built
  in the graph from the prompt length, so each compiled bucket serves every prompt that fits in it.
  GELU uses the exact erf formulation, as in PyTorch. The tanh approximation drifts across 28 layers.
* `prompt.zig`: prompt layout `[CLS] <type> question: … [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`,
  including the reference truncation rules. Calibration uses per-bucket temperatures clamped to [0.5, 5],
  like the reference runtime.
* `engine.zig`: compiles one executable per sequence-length bucket (128, 256, … up to `seqlen`), loads the weights once,
  and runs one forward pass per question in the smallest bucket that fits.

## Validation

On the sample inputs and on a plain-text state, the outputs match the MLX reference runtime
([laya-mlx](https://github.com/mizorewww/laya-mlx), f32) to all 4 printed decimals: probabilities, scores, confidence
and act probability. On an Apple M4 Pro (XLA CPU backend, f32), the 3 sample questions take about 1.6 s.
