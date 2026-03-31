## ✅ Diagnostic Pipeline: Next Actions Checklist

### Local Machine (macOS)

- [ ] Review the diagnostic pipeline overview:
  ```
  cat examples/ltx/DIAGNOSTIC_IMPLEMENTATION_SUMMARY.md
  ```

- [ ] Understand the pipeline flow:
  ```
  cat examples/ltx/DIAGNOSTIC_PIPELINE.md
  ```

- [ ] Quick reference for commands:
  ```
  cat examples/ltx/DIAGNOSTIC_QUICKSTART.md
  ```

- [ ] (Optional) Inspect the RoPE validation script:
  ```
  head -50 examples/ltx/diagnostic_rope_validation.py
  ```

### GPU Server (Remote)

**Prerequisites**: Ensure you have:
- Updated `replay_stage2_transformer_step.py`
- Updated `export_attention_fixture.py`
- New `diagnostic_rope_validation.py`

#### Step 1️⃣: Sync Updated Scripts
```bash
rsync -av ~/repos/zml/zml/examples/ltx/replay_stage2_transformer_step.py root@dev-oboulant:/root/repos/LTX-2/scripts/
rsync -av ~/repos/zml/zml/examples/ltx/export_attention_fixture.py root@dev-oboulant:/root/repos/LTX-2/scripts/
rsync -av ~/repos/zml/zml/examples/ltx/diagnostic_rope_validation.py root@dev-oboulant:/root/repos/LTX-2/scripts/
```

#### Step 2️⃣: Run Replay with Diagnostics
```bash
cd /root/repos/LTX-2
python3 scripts/replay_stage2_transformer_step.py \
    --pass-label diagnostic \
    --capture-kwargs \
    --capture-inputs \
    --include '^velocity_model\.transformer_blocks\.0\.attn1(\.|$)' \
    --distilled-lora-strength 0.0 \
    --step-idx 0
# Expected output: trace_run/acts_stage2_transformer_step_000_diagnostic.pt
# Look for lines: "kwargs hook registered for:" and "intermediate captured for:"
```

- [ ] **Verify**: Check for captured intermediates in output:
  ```bash
  grep "intermediate captured" /tmp/replay_output.log
  ```

#### Step 3️⃣: Export Fixture with Diagnostics
```bash
python3 scripts/export_attention_fixture.py \
    trace_run/acts_stage2_transformer_step_000_diagnostic.pt \
    /root/models/ltx-2.3/attn1_fixture_diagnostic.safetensors \
    --mode attn1
# Expected to see diagnostic keys in output
```

- [ ] **Verify**: Check fixture contains diagnostic keys:
  ```bash
  python3 -c "
  from safetensors.torch import load_file
  t = load_file('/root/models/ltx-2.3/attn1_fixture_diagnostic.safetensors')
  diagnostics = [k for k in t.keys() if 'diag' in k]
  print('Diagnostics found:', diagnostics)
  print('Total keys:', len(t.keys()))
  "
  ```

#### Step 4️⃣: Generate Python Reference Values
```bash
python3 scripts/diagnostic_rope_validation.py \
    /root/models/ltx-2.3/attn1_fixture_diagnostic.safetensors \
    /root/models/ltx-2.3/attn1_reference.safetensors \
    --attn-name attn1 \
    --num-heads 32 \
    --token-limit 256
# Expected output:
#   - Detected RoPE layout: split or interleaved
#   - q_head_split, k_head_split, v_head_split shapes
#   - q_rotated, k_rotated shapes
```

- [ ] **Verify**: Check all reference values were computed:
  ```bash
  python3 -c "
  from safetensors.torch import load_file
  t = load_file('/root/models/ltx-2.3/attn1_reference.safetensors')
  print('Reference keys:', sorted(t.keys()))
  print('All required keys present:', all(k in t.keys() for k in [
      'attn1.q_head_split', 'attn1.k_head_split', 'attn1.v_head_split',
      'attn1.q_rotated', 'attn1.k_rotated'
  ]))
  "
  ```

#### Step 5️⃣: Report Results

Share the following with development:

```bash
# 1. File sizes
ls -lh /root/models/ltx-2.3/attn1_fixture_diagnostic.safetensors
ls -lh /root/models/ltx-2.3/attn1_reference.safetensors

# 2. Fixture metadata
python3 -c "
from safetensors import safe_open
with safe_open('/root/models/ltx-2.3/attn1_fixture_diagnostic.safetensors', 'pt', '/root/models/ltx-2.3/') as f:
    print('Fixture metadata:', f.metadata())
"

# 3. Reference metadata
python3 -c "
from safetensors import safe_open
with safe_open('/root/models/ltx-2.3/attn1_reference.safetensors', 'pt', '/root/models/ltx-2.3/') as f:
    print('Reference metadata:', f.metadata())
"
```

### What This Validates

After completing these steps, the diagnostics file will contain ground-truth reference values for:

✅ **q_head_split**: [B=1, H=32, T=256, HD=128] — head-split query (reference)
✅ **k_head_split**: [B=1, H=32, T=256, HD=128] — head-split key (reference)
✅ **v_head_split**: [B=1, H=32, T=256, HD=128] — head-split value (reference)
✅ **q_rotated**: [B=1, H=32, T=256, HD=128] — query after RoPE (ground truth)
✅ **k_rotated**: [B=1, H=32, T=256, HD=128] — key after RoPE (ground truth)

### Next Phase: Zig Comparison

Once diagnostics are ready, the Zig checker will:
1. Load reference values
2. Compute same operations using ZML tensor library
3. Compare at each stage
4. Report which operation diverges from LTX

### Troubleshooting

**Q: Intermediates not captured?**
- A: Ensure `--capture-kwargs` AND `--include` regex both in command
- Check regex matches: `velocity_model.transformer_blocks.0.attn1`

**Q: Diagnostic keys missing from fixture?**
- A: Verify replay generated intermediate hooks output
- Grep for "intermediate captured for:" in replay logs

**Q: RoPE layout detection failed?**
- A: Check pe_cos final dimension (should be 64 for split, 128 for interleaved)
- Print: `python3 -c "from safetensors.torch import load_file; t = load_file('fixture'); print(t['attn1.pe_cos0'].shape)"`

**Q: Reference file is empty or has wrong shapes?**
- A: Verify fixture has q_head_split, k_head_split (not just q_diag0)
- Run diagnostic_rope_validation with `--token-limit 256` explicitly

### Success Criteria

✅ All 5 reference keys present in output file
✅ Shapes are [1, 32, 256, 128] (batch=1, heads=32, tokens=256, head_dim=128)
✅ Metadata shows `rope_layout: "split"` or `"interleaved"`
✅ No errors in diagnostic validation script

---

**Status**: Diagnostic pipeline implementation ✅ complete
**Next Action**: Execute steps 1️⃣-5️⃣ on GPU server
**Timeline**: ~5-10 minutes for full pipeline run
