#!/usr/bin/env python3
"""Open ASR Leaderboard evaluation for HiggsAudio3.

Evaluates HiggsAudio3 models on all 8 ESB benchmark datasets using the
official Whisper text normalizer, matching the HuggingFace leaderboard methodology.

Architecture: Whisper-Large-v3 encoder + Qwen decoder (1.7B or 8B).

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval.py [--datasets ami,earnings22,...] [--max-samples N]
    CUDA_VISIBLE_DEVICES=0 python eval.py --lora-path checkpoints/best

Environment variables:
    BOSON_PATH:  Path to boson-multimodal-ref library
    MODEL_PATH:  Path to HiggsAudio3 checkpoint (8B or 1.7B)
    WHISPER_PATH: Path to Whisper-Large-v3 processor
"""
import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------
BOSON_PATH = os.environ.get("BOSON_PATH", "boson-multimodal-ref")
sys.path.insert(0, BOSON_PATH)

from boson_multimodal.model.higgs_audio_3.configuration_higgs_audio import (
    HiggsAudio3Config, HiggsAudioEncoderConfig,
)
from boson_multimodal.model.higgs_audio_3.modeling_higgs_audio import HiggsAudio3Model
from transformers import AutoConfig, AutoModel, AutoTokenizer, WhisperProcessor

try:
    AutoConfig.register("higgs_audio_encoder", HiggsAudioEncoderConfig)
    AutoConfig.register("higgs_audio_3", HiggsAudio3Config)
    AutoModel.register(HiggsAudio3Config, HiggsAudio3Model)
except ValueError:
    pass

from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.data_types import ChatMLSample, AudioContent, Message
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample_qwen

# Whisper normalizer (official leaderboard normalization)
from whisper_normalizer.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "bosonai/higgs-audio-understanding-v3-8b")
WHISPER_PATH = os.environ.get("WHISPER_PATH", "openai/whisper-large-v3")
ESB_DATASET = "hf-audio/esb-datasets-test-only-sorted"

# The official leaderboard uses these 7 datasets (with librispeech split into clean+other = 8 WER values)
LEADERBOARD_DATASETS = [
    "ami",
    "earnings22",
    "gigaspeech",
    "librispeech",   # evaluated as test.clean + test.other separately
    "spgispeech",
    "tedlium",
    "voxpopuli",
]

# Best prompt from 1000+ experiments
USER_PROMPT = "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."
ENABLE_THINKING = True

# Generation parameters
MAX_NEW_TOKENS = 1024 if ENABLE_THINKING else 256
REPETITION_PENALTY = 1.0  # 1.0 = disabled (>1.0 hurts ASR by forcing hallucination)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def load_pipeline(device="cuda", lora_path=None, **kwargs):
    print(f"Loading model from {MODEL_PATH}...", flush=True)
    local_files = os.path.isdir(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=local_files)
    model = AutoModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map=device, local_files_only=local_files,
    )

    if lora_path is not None:
        print(f"Loading LoRA adapter from {lora_path}...", flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

        lora_scale = kwargs.get("lora_scale", 1.0)
        if lora_scale != 1.0:
            print(f"Scaling LoRA weights by {lora_scale}...", flush=True)
            scaled = 0
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.data = param.data * lora_scale
                    scaled += 1
            print(f"  Scaled {scaled} LoRA parameters.", flush=True)

        model = model.merge_and_unload()
        print(f"LoRA adapter merged (scale={lora_scale}).", flush=True)

    model = model.eval()
    dev = next(model.parameters()).device

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=local_files)
    model.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
    model.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    whisper_local = os.path.isdir(WHISPER_PATH)
    whisper_proc = WhisperProcessor.from_pretrained(WHISPER_PATH, local_files_only=whisper_local)

    collator = HiggsAudioSampleCollator(
        whisper_processor=whisper_proc,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        encode_whisper_embed=config.encode_whisper_embed,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=config.encode_audio_in_tokens,
        use_delay_pattern=config.use_delay_pattern,
        round_to=1,
        audio_num_codebooks=config.audio_num_codebooks,
        chunk_size_seconds=getattr(config, "chunk_size_seconds", 30),
        encoder_padding_method=getattr(config, "encoder_padding_method", "max_length"),
    )
    print(f"Model loaded on {dev}. Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)
    return {"model": model, "tokenizer": tokenizer, "collator": collator, "device": dev}


def transcribe(audio_np, pipeline):
    model = pipeline["model"]
    tokenizer = pipeline["tokenizer"]
    collator = pipeline["collator"]
    device = pipeline["device"]

    messages = [Message(role="user", content=[USER_PROMPT, AudioContent(audio_url="placeholder")])]
    chatml = ChatMLSample(messages=messages)
    prep_fn = partial(prepare_chatml_sample_qwen, enable_thinking=ENABLE_THINKING)
    input_tokens, _, _, _ = prep_fn(chatml, tokenizer, add_generation_prompt=True)

    sample = ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_tokens), label_ids=None,
        audio_ids_concat=None, audio_ids_start=None,
        audio_waveforms_concat=torch.tensor(audio_np, dtype=torch.float32),
        audio_waveforms_start=torch.tensor([0]),
        audio_sample_rate=torch.tensor([16000]),
        audio_speaker_indices=torch.tensor([0]),
    )

    batch = asdict(collator([sample]))
    batch = {k: v.to(device).contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "do_sample": False,
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
        "tokenizer": tokenizer,
    }
    if REPETITION_PENALTY > 1.0:
        gen_kwargs["repetition_penalty"] = REPETITION_PENALTY

    with torch.inference_mode():
        outputs = model.generate(**batch, **gen_kwargs)

    output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    parts = full_text.split("assistant\n")
    hyp = parts[-1] if len(parts) > 1 else full_text

    # Remove complete <think>...</think> blocks
    hyp = re.sub(r"<think>.*?</think>", "", hyp, flags=re.DOTALL)
    # Handle truncated thinking (</think> missing because max_new_tokens exhausted)
    if "<think>" in hyp:
        hyp = hyp[:hyp.index("<think>")].strip()
    # Remove special tokens
    hyp = re.sub(r"<\|.*?\|>", "", hyp)
    hyp = hyp.strip()

    # Post-processing: remove excessive word repetitions
    hyp = _fix_repetitions(hyp)
    return hyp


def _fix_repetitions(text, max_repeat=3):
    """Remove consecutive word repetitions beyond max_repeat."""
    words = text.split()
    if len(words) < max_repeat + 1:
        return text
    result = []
    for w in words:
        if result and result[-1] == w:
            count = 1
            for prev in reversed(result):
                if prev == w:
                    count += 1
                else:
                    break
            if count < max_repeat:
                result.append(w)
        else:
            result.append(w)
    return ' '.join(result)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def get_text(sample):
    for key in ["text", "sentence", "normalized_text", "transcript", "transcription"]:
        if key in sample:
            return sample[key]
    return ""


def evaluate_dataset(dataset_name, pipeline, max_samples=0, output_dir=None, skip_samples=0):
    from datasets import load_dataset, Audio

    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*60}", flush=True)

    # LibriSpeech: evaluate clean and other SEPARATELY for proper leaderboard scoring
    if dataset_name == "librispeech":
        results = {}
        for split_name in ["test.clean", "test.other"]:
            ds = load_dataset(ESB_DATASET, dataset_name, split=split_name)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            label = "clean" if "clean" in split_name else "other"
            print(f"\n  --- LibriSpeech {label} ({len(ds)} samples) ---", flush=True)

            if skip_samples > 0:
                end = min(skip_samples + max_samples, len(ds)) if max_samples > 0 else len(ds)
                ds = ds.select(range(skip_samples, end))
                print(f"  Skipped {skip_samples}, using {len(ds)} samples", flush=True)
            elif max_samples > 0:
                ds = ds.select(range(min(max_samples, len(ds))))
                print(f"  Truncated to {len(ds)} samples", flush=True)

            result = _evaluate_split(
                f"librispeech_{label}", ds, pipeline,
                output_dir=output_dir,
            )
            results[f"librispeech_{label}"] = result

        return results

    ds = load_dataset(ESB_DATASET, dataset_name, split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  Loaded {len(ds)} samples", flush=True)

    if skip_samples > 0:
        end = min(skip_samples + max_samples, len(ds)) if max_samples > 0 else len(ds)
        ds = ds.select(range(skip_samples, end))
        print(f"  Skipped {skip_samples}, using {len(ds)} samples", flush=True)
    elif max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"  Truncated to {len(ds)} samples", flush=True)

    result = _evaluate_split(dataset_name, ds, pipeline, output_dir=output_dir)
    return {dataset_name: result}


def _evaluate_split(split_name, ds, pipeline, output_dir=None):
    """Evaluate a single dataset split. Returns result dict."""
    import evaluate

    wer_metric = evaluate.load("wer")
    references = []
    predictions = []
    raw_refs = []
    raw_hyps = []
    total_audio_time = 0.0
    total_infer_time = 0.0

    # Checkpoint/resume support
    checkpoint_file = None
    start_idx = 0
    if output_dir is not None:
        checkpoint_file = Path(output_dir) / f"checkpoint_{split_name}.json"
        if checkpoint_file.exists():
            try:
                ckpt = json.loads(checkpoint_file.read_text())
                references = ckpt["references"]
                predictions = ckpt["predictions"]
                raw_refs = ckpt.get("raw_refs", [])
                raw_hyps = ckpt.get("raw_hyps", [])
                start_idx = ckpt["next_idx"]
                total_audio_time = ckpt.get("total_audio_time", 0.0)
                total_infer_time = ckpt.get("total_infer_time", 0.0)
                if start_idx < len(ds):
                    print(f"  Resuming {split_name} from sample {start_idx}/{len(ds)}", flush=True)
                else:
                    print(f"  Checkpoint complete for {split_name}", flush=True)
            except Exception as e:
                print(f"  Could not load checkpoint: {e}, starting fresh", flush=True)
                start_idx = 0

    skipped = 0
    for i in range(start_idx, len(ds)):
        sample = ds[i]
        ref_text = get_text(sample)
        if not ref_text.strip() or ref_text.strip() == "ignore time segment in scoring":
            skipped += 1
            continue

        audio = sample["audio"]
        audio_np = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        audio_duration = len(audio_np) / sr
        total_audio_time += audio_duration

        t0 = time.time()
        try:
            hyp = transcribe(audio_np, pipeline)
        except Exception as e:
            print(f"    Error sample {i}: {e}", flush=True)
            hyp = ""
        infer_time = time.time() - t0
        total_infer_time += infer_time

        # Apply official Whisper normalizer
        norm_ref = normalizer(ref_text)
        norm_hyp = normalizer(hyp)

        if norm_ref.strip():
            references.append(norm_ref)
            predictions.append(norm_hyp)
            raw_refs.append(ref_text)
            raw_hyps.append(hyp)

        if (i + 1) % 50 == 0 or (i + 1) == len(ds):
            interim_wer = wer_metric.compute(references=references, predictions=predictions)
            rtfx = total_audio_time / total_infer_time if total_infer_time > 0 else 0
            print(
                f"    {i+1}/{len(ds)} | WER={interim_wer*100:.2f}% | "
                f"RTFx={rtfx:.1f} | {total_infer_time:.0f}s",
                flush=True,
            )

        # Save checkpoint every 100 samples
        if checkpoint_file is not None and (i + 1) % 100 == 0:
            try:
                tmp = checkpoint_file.with_suffix(".tmp")
                tmp.write_text(json.dumps({
                    "references": references, "predictions": predictions,
                    "raw_refs": raw_refs, "raw_hyps": raw_hyps,
                    "next_idx": i + 1, "skipped": skipped,
                    "total_audio_time": total_audio_time,
                    "total_infer_time": total_infer_time,
                }))
                tmp.replace(checkpoint_file)
            except Exception:
                pass

    if not references:
        return {"wer": None, "rtfx": None, "count": 0}

    final_wer = wer_metric.compute(references=references, predictions=predictions)
    rtfx = total_audio_time / total_infer_time if total_infer_time > 0 else 0

    print(f"  FINAL {split_name}: WER={final_wer*100:.2f}% | RTFx={rtfx:.1f} | {len(references)} samples")

    # Save ref/hyp pairs for error analysis
    if output_dir is not None:
        outputs_file = Path(output_dir) / f"outputs_{split_name}.json"
        outputs_file.write_text(json.dumps({
            "references": references,
            "predictions": predictions,
            "raw_refs": raw_refs,
            "raw_hyps": raw_hyps,
            "wer": round(final_wer * 100, 2),
        }, indent=2))

    # Clean up checkpoint if completed
    if checkpoint_file is not None and checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
        except Exception:
            pass

    return {
        "wer": round(final_wer * 100, 2),
        "rtfx": round(rtfx, 1),
        "count": len(references),
        "audio_hours": round(total_audio_time / 3600, 2),
        "infer_time": round(total_infer_time, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(LEADERBOARD_DATASETS),
                        help="Comma-separated dataset names")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--skip-samples", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--output", default="results/leaderboard_results.json")
    parser.add_argument("--output-dir", default="results/leaderboard_outputs",
                        help="Directory for ref/hyp outputs and checkpoints")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--lora-path", default=None,
                        help="Path to LoRA adapter weights")
    parser.add_argument("--lora-scale", type=float, default=1.0,
                        help="Scale factor for LoRA weights (0-1, lower = more conservative)")
    parser.add_argument("--prompt", default=None,
                        help="Override the default transcription prompt")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    global ENABLE_THINKING, MAX_NEW_TOKENS, USER_PROMPT
    if args.no_thinking:
        ENABLE_THINKING = False
        MAX_NEW_TOKENS = 256
    if args.prompt:
        USER_PROMPT = args.prompt
        print(f"Using custom prompt: {USER_PROMPT}", flush=True)

    datasets_list = args.datasets.split(",")
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(lora_path=args.lora_path, lora_scale=args.lora_scale)

    all_results = {}
    all_wers = []

    for ds_name in datasets_list:
        split_results = evaluate_dataset(ds_name, pipeline, args.max_samples,
                                         output_dir=output_dir,
                                         skip_samples=args.skip_samples)
        for split_name, result in split_results.items():
            all_results[split_name] = result
            if result["wer"] is not None:
                all_wers.append(result["wer"])

    avg_wer = round(sum(all_wers) / len(all_wers), 2) if all_wers else None
    avg_rtfx = round(
        sum(r["rtfx"] for r in all_results.values() if r.get("rtfx"))
        / max(1, sum(1 for r in all_results.values() if r.get("rtfx"))),
        1,
    ) if all_wers else None

    summary = {
        "avg_wer": avg_wer,
        "avg_rtfx": avg_rtfx,
        "per_dataset": all_results,
        "model": MODEL_PATH,
        "prompt": USER_PROMPT,
        "thinking": ENABLE_THINKING,
        "max_new_tokens": MAX_NEW_TOKENS,
        "normalizer": "EnglishTextNormalizer (Whisper)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"LEADERBOARD RESULTS")
    print(f"{'='*60}")
    print(f"Average WER: {avg_wer}% (over {len(all_wers)} splits)")
    print(f"Average RTFx: {avg_rtfx}")
    print(f"\nPer-dataset:")
    for name, r in sorted(all_results.items()):
        if r["wer"] is not None:
            print(f"  {name:>20}: {r['wer']:6.2f}% WER | RTFx {r['rtfx']} | {r['count']} samples")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
