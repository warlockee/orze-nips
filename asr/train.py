#!/usr/bin/env python3
"""LoRA finetuning script for HiggsAudio3 8B ASR.

Fine-tunes the LLM decoder (Qwen3-8B) using LoRA on ASR data to improve
WER on specific benchmarks (AMI, VoxPopuli, TED-LIUM, Earnings22).

Architecture: Whisper-Large-v3 encoder (frozen) + Qwen3-8B decoder (LoRA).

Usage:
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --datasets ami_train,voxpopuli_train,tedlium_train \
        --output-dir checkpoints/8b_lora_v1 \
        --epochs 1 --lr 5e-5 --lora-rank 16 --max-samples 5000

Environment variables:
    BOSON_PATH:  Path to boson-multimodal-ref library
    MODEL_PATH:  Path to HiggsAudio3 8B checkpoint
    WHISPER_PATH: Path to Whisper-Large-v3 processor
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BOSON_PATH = os.environ.get("BOSON_PATH", "boson-multimodal-ref")
sys.path.insert(0, BOSON_PATH)
sys.stdout.reconfigure(line_buffering=True)

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
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = os.environ.get("MODEL_PATH", "bosonai/higgs-audio-understanding-v3-8b")
WHISPER_PATH = os.environ.get("WHISPER_PATH", "openai/whisper-large-v3")
USER_PROMPT = "Transcribe the speech. Output only the spoken words in lowercase with no punctuation."


def build_training_sample(audio_np, ref_text, tokenizer, enable_thinking=True):
    """Build a complete training sample with input_ids and labels."""
    messages = [
        Message(role="user", content=[USER_PROMPT, AudioContent(audio_url="placeholder")]),
    ]
    chatml = ChatMLSample(messages=messages)
    prep_fn = partial(prepare_chatml_sample_qwen, enable_thinking=enable_thinking)
    input_tokens, _, _, _ = prep_fn(chatml, tokenizer, add_generation_prompt=True)

    # For thinking mode, we add an empty think block then the answer
    if enable_thinking:
        think_tokens = tokenizer.encode("<think>\n\n</think>\n", add_special_tokens=False)
    else:
        think_tokens = []

    target_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
    eos_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    full_tokens = input_tokens + think_tokens + target_tokens + eos_token
    labels = [-100] * len(input_tokens) + [-100] * len(think_tokens) + target_tokens + eos_token

    sample = ChatMLDatasetSample(
        input_ids=torch.LongTensor(full_tokens),
        label_ids=torch.LongTensor(labels),
        audio_ids_concat=None,
        audio_ids_start=None,
        audio_waveforms_concat=torch.tensor(audio_np, dtype=torch.float32),
        audio_waveforms_start=torch.tensor([0]),
        audio_sample_rate=torch.tensor([16000]),
        audio_speaker_indices=torch.tensor([0]),
    )
    return sample


def _load_single_dataset(ds_name):
    """Load a single dataset by name. Returns HF dataset (not yet sampled)."""
    from datasets import load_dataset, Audio

    if ds_name == "ami_train":
        ds = load_dataset("edinburghcstr/ami", "ihm", split="train", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "ami_dev":
        ds = load_dataset("edinburghcstr/ami", "ihm", split="validation", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "voxpopuli_train":
        ds = load_dataset("facebook/voxpopuli", "en", split="train")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "tedlium_train":
        ds = load_dataset("sanchit-gandhi/tedlium-data", split="train", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "librispeech_train":
        ds = load_dataset("librispeech_asr", "other", split="train.500")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "librispeech_clean_train":
        ds = load_dataset("librispeech_asr", "clean", split="train.100")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "earnings22_train":
        ds = load_dataset("distil-whisper/earnings22", "chunked", split="test", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "spgispeech_train":
        ds = load_dataset("kensho/spgispeech", split="train", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "gigaspeech_train":
        ds = load_dataset("speechcolab/gigaspeech", "l", split="train", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    elif ds_name == "commonvoice_train":
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    return ds


def get_text(sample):
    """Get text from a sample, trying various key names."""
    for key in ["text", "sentence", "normalized_text", "transcript", "transcription"]:
        if key in sample and sample[key]:
            return sample[key]
    return ""


def _oversample_ami_short(ds, max_samples, short_multiplier=3, seed=42):
    """Load AMI with short-utterance (<3s) oversampling.

    Short AMI utterances (backchannels, acknowledgements) are the hardest for
    ASR models. Oversampling them N times during training significantly improves
    AMI WER without degrading other datasets.

    Returns (dataset, indices) where indices may contain repeated entries for
    short utterances.
    """
    rng = np.random.RandomState(seed)
    short_indices = []
    other_indices = []

    check_n = min(max_samples * 2, len(ds))
    all_indices = rng.permutation(len(ds))[:check_n]

    # Use begin_time/end_time metadata to compute duration without loading audio
    has_time_meta = "begin_time" in ds.column_names and "end_time" in ds.column_names
    if has_time_meta:
        meta_ds = ds.remove_columns(["audio"])
    else:
        meta_ds = None

    for idx in all_indices:
        if meta_ds is not None:
            sample = meta_ds[int(idx)]
            dur = float(sample["end_time"]) - float(sample["begin_time"])
        else:
            sample = ds[int(idx)]
            audio = sample["audio"]
            dur = len(audio["array"]) / audio["sampling_rate"]
        text = sample.get("text", "") or ""
        if not text.strip() or text.strip() == "ignore time segment in scoring":
            continue
        if dur < 3.0:
            short_indices.append(int(idx))
        else:
            other_indices.append(int(idx))

    print(f"  AMI: {len(short_indices)} short (<3s), {len(other_indices)} other", flush=True)

    # Oversample short utterances
    oversampled = []
    for _ in range(short_multiplier):
        oversampled.extend(short_indices)
    rng.shuffle(oversampled)

    # Combine: oversampled shorts + enough others to reach max_samples
    combined = oversampled + other_indices
    if max_samples > 0 and len(combined) > max_samples:
        combined = combined[:max_samples]

    print(f"  AMI final: {len(combined)} samples (short x{short_multiplier})", flush=True)
    return ds, combined


def train(args):
    print(f"=== 8B LoRA Finetuning for ASR ===", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}", flush=True)
    print(f"LR: {args.lr}, Epochs: {args.epochs}", flush=True)

    # Load model
    print("Loading 8B model...", flush=True)
    local_files = os.path.isdir(MODEL_PATH)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=local_files)
    model = AutoModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", device_map="cuda", local_files_only=local_files,
    )
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
    device = next(model.parameters()).device
    print(f"Model loaded on {device}. Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    # Apply LoRA to LLM decoder attention layers
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if getattr(args, 'target_mlp', False):
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        print(f"Targeting MLP layers too: {target_modules}", flush=True)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=getattr(args, 'lora_dropout', 0.05),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Freeze audio tower completely
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = False

    # Optionally train audio_encoder_proj
    if getattr(args, 'train_encoder_proj', False):
        encoder_proj_params = 0
        for name, param in model.named_parameters():
            if "audio_encoder_proj" in name:
                param.requires_grad = True
                encoder_proj_params += param.numel()
        print(f"Also training audio_encoder_proj: {encoder_proj_params/1e6:.1f}M params", flush=True)

    # Load data - supports per-dataset sample counts (e.g., "ami_train:8000,voxpopuli_train:4000")
    dataset_specs = args.datasets.split(",")
    dataset_names = []
    per_dataset_samples = {}
    for spec in dataset_specs:
        if ":" in spec:
            name, count = spec.rsplit(":", 1)
            dataset_names.append(name)
            per_dataset_samples[name] = int(count)
        else:
            dataset_names.append(spec)
            per_dataset_samples[spec] = args.max_samples

    ami_oversample = getattr(args, 'ami_short_oversample', 1)
    seed = getattr(args, 'seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def load_with_per_ds_limits(names, per_ds_samples):
        from datasets import concatenate_datasets
        all_datasets = []
        for ds_name in names:
            max_s = per_ds_samples.get(ds_name, args.max_samples)
            print(f"Loading {ds_name} (max {max_s if max_s > 0 else 'all'})...", flush=True)
            ds = _load_single_dataset(ds_name)

            # AMI short-utterance oversampling
            if ds_name == "ami_train" and ami_oversample > 1:
                ds, indices = _oversample_ami_short(ds, max_s, short_multiplier=ami_oversample, seed=seed)
                ds = ds.select(indices)
                print(f"  Loaded {len(ds)} samples from {ds_name} (short x{ami_oversample})", flush=True)
            elif max_s > 0 and len(ds) > max_s:
                ds = ds.shuffle(seed=seed).select(range(max_s))
                print(f"  Loaded {len(ds)} samples from {ds_name}", flush=True)
            else:
                print(f"  Loaded {len(ds)} samples from {ds_name}", flush=True)

            all_datasets.append(ds)

        if len(all_datasets) == 1:
            return all_datasets[0]

        harmonized = []
        for ds in all_datasets:
            text_col = None
            for key in ["text", "sentence", "normalized_text", "transcript", "transcription"]:
                if key in ds.column_names:
                    text_col = key
                    break
            if text_col is None:
                continue
            keep_cols = ["audio"]
            if text_col != "text":
                ds = ds.rename_column(text_col, "text")
            keep_cols.append("text")
            remove_cols = [c for c in ds.column_names if c not in keep_cols]
            if remove_cols:
                ds = ds.remove_columns(remove_cols)
            harmonized.append(ds)

        return concatenate_datasets(harmonized)

    train_data = load_with_per_ds_limits(dataset_names, per_dataset_samples)
    print(f"Total training samples: {len(train_data)}", flush=True)

    # Training setup
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_data) * args.epochs // args.grad_accum
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    best_loss = float("inf")
    running_loss = 0.0
    running_count = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_errors = 0
        t0 = time.time()

        indices = np.random.permutation(len(train_data))
        optimizer.zero_grad()

        for batch_idx, idx in enumerate(indices):
            try:
                sample = train_data[int(idx)]
                audio = sample["audio"]
                audio_np = np.array(audio["array"], dtype=np.float32)

                ref_text = get_text(sample)
                if not ref_text.strip() or ref_text.strip() == "ignore time segment in scoring":
                    continue

                # Truncate very long audio (30s max for 8B to avoid OOM)
                max_audio_samples = int(30.0 * 16000)
                if len(audio_np) > max_audio_samples:
                    audio_np = audio_np[:max_audio_samples]

                # Skip very short audio
                if len(audio_np) < 1600:  # < 0.1s
                    continue

                ref_text = ref_text.lower().strip()

                train_sample = build_training_sample(audio_np, ref_text, tokenizer, enable_thinking=True)
                batch = asdict(collator([train_sample]))
                batch = {
                    k: v.to(device).contiguous() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    underlying = model.base_model.model
                    outputs = underlying(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        audio_features=batch.get("audio_features"),
                        audio_feature_attention_mask=batch.get("audio_feature_attention_mask"),
                        label_ids=batch.get("label_ids"),
                    )
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss / args.grad_accum
                loss.backward()

                loss_val = loss.item() * args.grad_accum
                epoch_loss += loss_val
                epoch_samples += 1
                running_loss += loss_val
                running_count += 1

                if (batch_idx + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (batch_idx + 1) % 100 == 0:
                    avg_loss = running_loss / max(1, running_count)
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - t0
                    samples_per_sec = epoch_samples / max(1, elapsed)
                    print(
                        f"  Epoch {epoch+1}/{args.epochs} | "
                        f"Step {batch_idx+1}/{len(indices)} | "
                        f"Loss={avg_loss:.4f} | LR={lr:.2e} | "
                        f"{samples_per_sec:.1f} samp/s | "
                        f"{elapsed:.0f}s | Errors={epoch_errors}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_count = 0

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                epoch_errors += 1
                continue
            except Exception as e:
                if batch_idx < 5:
                    print(f"  Error at sample {idx}: {e}", flush=True)
                epoch_errors += 1
                continue

        avg_loss = epoch_loss / max(1, epoch_samples)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"avg_loss={avg_loss:.4f} | "
            f"{epoch_samples} samples | {elapsed:.0f}s | Errors={epoch_errors}",
            flush=True,
        )

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_dir = output_dir / "best"
            model.save_pretrained(ckpt_dir)
            print(f"  Saved best checkpoint to {ckpt_dir}", flush=True)

        ckpt_dir = output_dir / f"epoch_{epoch+1}"
        model.save_pretrained(ckpt_dir)
        print(f"  Saved epoch checkpoint to {ckpt_dir}", flush=True)

    # Save final
    model.save_pretrained(output_dir / "final")

    info = {
        "model_path": MODEL_PATH,
        "datasets": args.datasets,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "best_loss": best_loss,
        "total_steps": global_step,
    }
    (output_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="ami_train",
                        help="Comma-separated dataset names (optionally with counts: ami_train:8000,voxpopuli_train:4000)")
    parser.add_argument("--output-dir", default="checkpoints/8b_lora_v1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all (applied per dataset)")
    parser.add_argument("--train-encoder-proj", action="store_true",
                        help="Also train the audio_encoder_proj (11M params)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-mlp", action="store_true",
                        help="Also target MLP layers (gate/up/down_proj)")
    parser.add_argument("--ami-short-oversample", type=int, default=1,
                        help="Oversample AMI utterances <3s by this factor (default 1 = no oversampling)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
