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

## Interactive demo

```bash
bazel run //examples/laya -- --model=hf://convaiinnovations/laya --serve=9000
```

Then open <http://localhost:9000>. The page is embedded in the binary (`demo.html`) and has no external dependencies.

* **Presets**: support ticket, product review, moderation and agent routing examples.
* **Question editor**: add `choice` / `score` / yes-no questions, type an option and press Enter, or click `{ } JSON`
  to edit the raw questions.
* **Decide** (or <kbd>⌘</kbd>/<kbd>Ctrl</kbd>+<kbd>Enter</kbd>): an animated pipeline follows the pass through
  tokenize, ModernBERT, decision head and calibration. Then probability bars, confidence rings, the score needle and
  the act/escalate flag animate in.
* **What the model read**: expands the exact prompt tokens with the `[MASK]` option markers highlighted.
* **Interaction guide**: a 5-step spotlight tour runs on the first visit. Reopen it with `? Guide` or <kbd>?</kbd>,
  navigate with <kbd>←</kbd>/<kbd>→</kbd> and close it with <kbd>Esc</kbd>.

The server also exposes the API directly:

```bash
curl -s localhost:9000/v1/decide -H 'content-type: application/json' -d '{
  "state": "The app crashes every time I open settings.",
  "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "engineering", "sales"]}}
}'
```

The response is the CLI output plus `prompts`, which lists the decoded tokens and marker positions of each question.
`GET /v1/health` reports the platform and sequence length. The server listens on `127.0.0.1` by default; use
`--host=0.0.0.0` to expose it.

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
