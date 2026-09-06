"""Compare --dump-dir artifacts against the upstream CPU implementation.

The reference repository and its Python dependencies are only needed for this
validation script. Both Zig binaries run independently of Python.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch


def compare(name, actual, expected, tolerance):
    actual = np.asarray(actual, np.float32).reshape(-1)
    expected = expected.detach().float().cpu().numpy().reshape(-1)
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shapes differ: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{name}: nonfinite values")
    relative = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
    cosine = np.dot(actual, expected) / max(np.linalg.norm(actual) * np.linalg.norm(expected), 1e-12)
    print(f"{name}: relative L2={relative:.6g}, cosine={cosine:.8f}", flush=True)
    if relative > tolerance:
        raise AssertionError(f"{name}: relative error exceeds {tolerance}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", default="Speak clearly and naturally.")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    args = parser.parse_args()
    if bool(args.ref_audio) != bool(args.ref_text):
        parser.error("--ref-audio and --ref-text must be supplied together")
    sys.path.insert(0, str(args.reference_repo.resolve()))
    from breeze_infer.runtime import load_runtime
    from breeze_infer.templates import get_template, prepare_inputs

    torch.set_num_threads(8)
    tokenizer, model, codec = load_runtime(args.model, device="cpu", attn_implementation="eager")
    request = {"text": args.text, "instruction": args.instruction, "speaker": "S0"}
    if args.ref_audio:
        request.update(ref_audio_path=str(args.ref_audio), ref_text=args.ref_text)
    inputs = prepare_inputs(tokenizer, codec, model, [request],
                            get_template("ref_edit_tata" if args.ref_audio else "tts_instruction"),
                            guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None)
    root = args.dump_dir

    def read(name, bf16=False):
        if bf16:
            return (np.fromfile(root / f"{name}.bf16", np.uint16).astype(np.uint32) << 16).view(np.float32)
        return np.fromfile(root / f"{name}.f32", np.float32)

    with torch.no_grad():
        prompt, _ = model.convert_input_ids_to_embeds(inputs["input_ids"], inputs["text_ids_mask"], inputs["text_ids_len"])
        if args.ref_audio:
            ref_codes = inputs["input_values"]
            native = np.fromfile(root / "reference_codes.u32", np.uint32).reshape(-1, 16)
            reference = ref_codes.cpu().numpy().reshape(-1, 16)
            agreement = np.mean(native == reference)
            print(f"Reference codec token agreement: {agreement:.6f}", flush=True)
            # A near-tie in one residual codebook can change all later codes in
            # that frame. Check the latent and each chosen centroid directly.
            import soundfile as sf
            wave, rate = sf.read(args.ref_audio, dtype="float32", always_2d=True)
            if rate != 24000:
                raise AssertionError("Native reference input must be 24 kHz")
            encoder = codec.model.encoder
            encoded = encoder.encoder(torch.from_numpy(wave.mean(axis=1))[None, None])
            encoded = encoder.encoder_transformer(encoded.transpose(1, 2), use_cache=False).last_hidden_state.transpose(1, 2)
            latent = encoder.downsample(encoded).transpose(1, 2)
            native_latent = read("reference_latent").reshape(1, -1, 512)
            compare("reference latent", native_latent, latent, 0.005)
            native_ids = torch.from_numpy(native.astype(np.int64))
            max_excess = 0.0
            for group, begin, count in [
                (encoder.quantizer.semantic_residual_vector_quantizer, 0, 1),
                (encoder.quantizer.acoustic_residual_vector_quantizer, 1, 15),
            ]:
                residual = group.input_proj(torch.from_numpy(native_latent).transpose(1, 2)).transpose(1, 2)[0]
                for i in range(count):
                    book = group.layers[i].codebook.embed
                    distances = torch.cdist(residual[None], book[None])[0]
                    chosen = native_ids[:, begin + i]
                    selected = distances.gather(1, chosen[:, None])[:, 0]
                    best = distances.min(dim=1).values
                    max_excess = max(max_excess, ((selected - best) / best.clamp_min(1e-8)).max().item())
                    residual = residual - book[chosen]
            print(f"Reference quantizer maximum relative distance excess: {max_excess:.6g}", flush=True)
            if max_excess > 0.001:
                raise AssertionError("Reference quantizer chose a non-nearest centroid beyond numerical tolerance")
            ref_codes = native_ids.unsqueeze(0)
            prompt[inputs["input_ids"] == model.config.audio_token_id] = model.backbone_model.embed_tokens(ref_codes).reshape(-1, 2048)
            prompt[inputs["input_ids"] == model.config.audio_eos_token_id] = model.backbone_model.embed_tokens(torch.zeros((1, 1, 16), dtype=torch.long)).reshape(-1, 2048)
        compare("prompt", read("prompt", True), prompt, 0.04)
        hidden = model.backbone_model(inputs_embeds=prompt, use_cache=False).last_hidden_state[:, -1:]
        compare("hidden", read("hidden", True), hidden, 0.04)
        compare("logits", read("logits"), model.lm_head(hidden), 0.04)
        codes = torch.from_numpy(np.fromfile(root / "codes.u32", np.uint32).astype(np.int64).reshape(-1, 16))
        if (root / "depth_logits.f32").exists():
            depth_ids = torch.tensor([[0, codes[0, 0].item()]])
            depth_hidden = model.depth_decoder.model(input_ids=depth_ids, backbone_last_hidden_state=hidden[:, 0], use_cache=False).last_hidden_state[:, -1:]
            depth_logits = depth_hidden @ model.depth_decoder.codebooks_head.weight[0]
            compare("depth logits", read("depth_logits"), depth_logits, 0.06)
        if (root / "decode_logits.f32").exists():
            extended = torch.cat([prompt, model.backbone_model.embed_tokens(codes[:1].unsqueeze(0))], dim=1)
            decoded = model.backbone_model(inputs_embeds=extended, use_cache=False).last_hidden_state[:, -1:]
            compare("cached decode logits", read("decode_logits"), model.lm_head(decoded), 0.06)
        wav, _ = codec.decode([{"audio_codes": codes}])
        compare("waveform", read("audio"), torch.as_tensor(wav[0]), 0.01)
    print("All reference comparisons passed.")


if __name__ == "__main__":
    main()
