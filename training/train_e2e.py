#!/usr/bin/env python3
"""Nexar Collision Detection — End-to-End Frame-Level Fine-Tuning (Recipe B).

MViTv2-S pretrained on Kinetics-400, fine-tuned with frame-level sampling
and weighted random sampling (alert-zone upweighting). 1st-place Kaggle
solution approach achieving ~0.898 mAP.

Input: (B, 3, 16, 224, 224) clips (resize 256, center crop 224).
Output: binary collision probability per frame, aggregated per video at eval.
"""

import argparse
import gc
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:
    print("ERROR: PyTorch is required. Install with:")
    print("  pip install torch")
    sys.exit(1)

try:
    import torchvision
    from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
except ImportError:
    print("ERROR: torchvision is required. Install with:")
    print("  pip install torchvision")
    sys.exit(1)

try:
    import av
except ImportError:
    print("ERROR: PyAV is required for video decoding. Install with:")
    print("  pip install av")
    sys.exit(1)

try:
    from sklearn.metrics import average_precision_score
except ImportError:
    print("ERROR: scikit-learn is required. Install with:")
    print("  pip install scikit-learn")
    sys.exit(1)

import csv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
# MViTv2-S Kinetics-400 normalization (NOT standard ImageNet)
MVIT_MEAN = [0.45, 0.45, 0.45]
MVIT_STD = [0.225, 0.225, 0.225]
RESIZE_SIZE = 256  # resize short side to this
CROP_SIZE = 224    # center crop to this

TRAIN_VIDEO_DIR = os.environ.get("NEXAR_TRAIN_DIR", "data/train")
TRAIN_CSV = os.environ.get("NEXAR_TRAIN_CSV", "data/train.csv")
TEST_VIDEO_DIR = os.environ.get("NEXAR_TEST_DIR", "data/test")
TEST_CSV = os.environ.get("NEXAR_TEST_CSV", "data/solution.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def get_video_info(video_path):
    """Return (num_frames, fps) for a video file."""
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        # frames attribute can be 0 for some containers; fallback to duration
        n_frames = stream.frames
        if n_frames == 0 and stream.duration and stream.time_base:
            n_frames = int(float(stream.duration * stream.time_base) * fps)
        return n_frames, fps
    finally:
        container.close()


class _VideoDecodeTimeout(Exception):
    pass


def _decode_alarm_handler(signum, frame):
    raise _VideoDecodeTimeout()


def decode_video_frames(video_path, frame_indices, timeout_sec=30):
    """Decode specific frames from a video using PyAV with seeking.

    Args:
        video_path: path to mp4
        frame_indices: sorted list of 0-based frame indices to decode
        timeout_sec: max seconds before aborting decode (0=no timeout)

    Returns:
        dict mapping frame_index -> np.ndarray (H, W, 3) uint8
    """
    if not frame_indices:
        return {}

    # Set decode timeout (main thread only)
    use_alarm = timeout_sec > 0
    if use_alarm:
        try:
            old_handler = signal.signal(signal.SIGALRM, _decode_alarm_handler)
            signal.alarm(timeout_sec)
        except (ValueError, OSError):
            use_alarm = False  # Not main thread (training DataLoader workers)

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        # NOTE: Do NOT set stream.thread_type = "AUTO" — FFmpeg internal
        # decoder threads deadlock on NFS (futex_wait_queue). Single-threaded
        # decode is slower but never hangs.
        fps = float(stream.average_rate) if stream.average_rate else 30.0

        frames_needed = set(frame_indices)
        result = {}

        # Seek to slightly before the earliest needed frame to avoid
        # decoding from the start of long videos
        min_frame = min(frame_indices)
        if min_frame > 30:
            # Seek to ~1 second before the earliest frame
            seek_frame = max(0, min_frame - int(fps))
            seek_ts = int(seek_frame / fps / stream.time_base)
            container.seek(seek_ts, stream=stream)

        frame_idx = -1
        for frame in container.decode(video=0):
            # After seeking, we need to figure out which frame index we're at.
            # Use PTS to compute frame index.
            if frame.pts is not None and stream.time_base:
                frame_idx = int(float(frame.pts * stream.time_base) * fps + 0.5)
            else:
                frame_idx += 1

            if frame_idx in frames_needed:
                result[frame_idx] = frame.to_ndarray(format="rgb24")
                frames_needed.discard(frame_idx)
                if not frames_needed:
                    break

            # Safety: if we've gone past all needed frames, stop
            if frame_idx > max(frame_indices) + 5:
                break

        return result
    except _VideoDecodeTimeout:
        return result  # Return whatever frames we decoded before timeout
    finally:
        container.close()
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def load_clip(video_path, center_frame, num_frames=16, stride=4,
              n_total=None):
    """Load a 16-frame clip ending at center_frame.

    Frames are sampled going backward with the given stride:
    [center - (num_frames-1)*stride, ..., center - stride, center]

    Preprocessing matches MViT_V2_S_Weights.KINETICS400_V1:
      resize short side to 256, center crop 224x224,
      normalize with mean=0.45, std=0.225.

    Returns:
        torch.Tensor of shape (3, T, 224, 224)
    """
    if n_total is None:
        n_total, _ = get_video_info(video_path)

    # Compute frame indices going backward from center_frame
    indices = []
    for i in range(num_frames):
        idx = center_frame - (num_frames - 1 - i) * stride
        indices.append(idx)

    # Clamp to valid range
    indices = [max(0, min(idx, n_total - 1)) for idx in indices]

    # Decode needed unique frames
    unique_indices = sorted(set(indices))
    decoded = decode_video_frames(video_path, unique_indices)

    # If some frames failed to decode, use nearest available
    available = sorted(decoded.keys())
    if not available:
        return torch.zeros(3, num_frames, CROP_SIZE, CROP_SIZE)

    frames = []
    for idx in indices:
        if idx in decoded:
            frames.append(decoded[idx])
        else:
            nearest = min(available, key=lambda x: abs(x - idx))
            frames.append(decoded[nearest])

    # Stack -> (T, H, W, 3) uint8
    clip = np.stack(frames, axis=0)
    clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float()  # (T, 3, H, W)

    # Resize short side to RESIZE_SIZE, preserving aspect ratio
    _, _, h, w = clip.shape
    if h < w:
        new_h = RESIZE_SIZE
        new_w = int(w * RESIZE_SIZE / h)
    else:
        new_w = RESIZE_SIZE
        new_h = int(h * RESIZE_SIZE / w)
    clip = F.interpolate(clip, size=(new_h, new_w), mode="bilinear",
                         align_corners=False)

    # Center crop to CROP_SIZE x CROP_SIZE
    _, _, h, w = clip.shape
    top = (h - CROP_SIZE) // 2
    left = (w - CROP_SIZE) // 2
    clip = clip[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]

    # Normalize: [0,255] -> [0,1] -> MViT norm
    clip = clip / 255.0
    mean = torch.tensor(MVIT_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(MVIT_STD).view(1, 3, 1, 1)
    clip = (clip - mean) / std

    # Reshape to (3, T, H, W) for MViT
    clip = clip.permute(1, 0, 2, 3)  # (3, T, H, W)
    return clip


def load_clip_pair(video_path, center_frame, num_frames=16, stride=4,
                   n_total=None, step=4):
    """Load two temporally consecutive clips efficiently (single video open).

    clip_t is centered on center_frame, clip_t_next is centered on
    center_frame + step.  Both clips are decoded in one pass to minimise I/O.

    Returns:
        (clip_t, clip_t_next) — each (3, T, H, W)
    """
    if n_total is None:
        n_total, _ = get_video_info(video_path)

    def _indices(center):
        return [max(0, min(center - (num_frames - 1 - i) * stride, n_total - 1))
                for i in range(num_frames)]

    indices_t = _indices(center_frame)
    center_next = min(center_frame + step, n_total - 1)
    indices_next = _indices(center_next)

    all_indices = sorted(set(indices_t + indices_next))
    decoded = decode_video_frames(video_path, all_indices)
    available = sorted(decoded.keys()) if decoded else []

    def _build_clip(indices):
        if not available:
            return torch.zeros(3, num_frames, CROP_SIZE, CROP_SIZE)
        frames = []
        for idx in indices:
            if idx in decoded:
                frames.append(decoded[idx])
            else:
                nearest = min(available, key=lambda x: abs(x - idx))
                frames.append(decoded[nearest])
        clip = np.stack(frames, axis=0)
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        _, _, h, w = clip.shape
        if h < w:
            new_h, new_w = RESIZE_SIZE, int(w * RESIZE_SIZE / h)
        else:
            new_w, new_h = RESIZE_SIZE, int(h * RESIZE_SIZE / w)
        clip = F.interpolate(clip, size=(new_h, new_w), mode="bilinear",
                             align_corners=False)
        _, _, h, w = clip.shape
        top = (h - CROP_SIZE) // 2
        left = (w - CROP_SIZE) // 2
        clip = clip[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]
        clip = clip / 255.0
        mean = torch.tensor(MVIT_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(MVIT_STD).view(1, 3, 1, 1)
        clip = (clip - mean) / std
        clip = clip.permute(1, 0, 2, 3)
        return clip

    return _build_clip(indices_t), _build_clip(indices_next)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def load_train_annotations(csv_path):
    """Load train.csv -> list of dicts with id, target, time_of_event, time_of_alert."""
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "id": row["id"],
                "target": int(row["target"]),
                "time_of_event": float(row["time_of_event"]) if row["time_of_event"] else None,
                "time_of_alert": float(row["time_of_alert"]) if row["time_of_alert"] else None,
            }
            records.append(rec)
    return records


def load_test_annotations(csv_path):
    """Load solution.csv -> list of dicts with id, target."""
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "id": row["id"],
                "target": int(row["target"]),
            })
    return records


# ---------------------------------------------------------------------------
# Frame-Level Dataset
# ---------------------------------------------------------------------------

class FrameLevelDataset(Dataset):
    """Dataset that returns individual frame clips from pre-built frame list."""

    def __init__(self, frame_list, video_dir, num_frames=16, stride=4,
                 augment=False, augment_color=False, ffr=False):
        self.frame_list = frame_list
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.stride = stride
        self.augment = augment
        self.augment_color = augment_color
        self.ffr = ffr  # return clip pair for Future-Frame Regularization

    def __len__(self):
        return len(self.frame_list)

    def _augment_clip(self, clip):
        if self.augment:
            if random.random() < 0.5:
                clip = clip.flip(-1)
            if self.augment_color:
                if random.random() < 0.8:
                    brightness = 1.0 + random.uniform(-0.3, 0.3)
                    contrast = 1.0 + random.uniform(-0.3, 0.3)
                    clip = clip * contrast + (brightness - 1.0) * 0.5
                if random.random() < 0.2:
                    gray = clip.mean(dim=0, keepdim=True)
                    clip = gray.expand_as(clip)
        return clip

    def __getitem__(self, idx):
        rec = self.frame_list[idx]
        video_path = os.path.join(self.video_dir, f"{rec['video_id']}.mp4")
        label = rec["label"]

        if self.ffr:
            try:
                clip_t, clip_next = load_clip_pair(
                    video_path, rec["frame_idx"],
                    num_frames=self.num_frames, stride=self.stride,
                    n_total=rec.get("n_frames"), step=self.stride)
            except Exception:
                z = torch.zeros(3, self.num_frames, CROP_SIZE, CROP_SIZE)
                clip_t, clip_next = z, z.clone()
            clip_t = self._augment_clip(clip_t)
            clip_next = self._augment_clip(clip_next)
            return clip_t, clip_next, torch.tensor(label, dtype=torch.float32)

        try:
            clip = load_clip(video_path, rec["frame_idx"],
                             num_frames=self.num_frames, stride=self.stride,
                             n_total=rec.get("n_frames"))
        except Exception:
            clip = torch.zeros(3, self.num_frames, CROP_SIZE, CROP_SIZE)

        clip = self._augment_clip(clip)
        return clip, torch.tensor(label, dtype=torch.float32)


class VideoEvalDataset(Dataset):
    """Dataset for video-level evaluation: sample every Nth frame as center."""

    def __init__(self, video_ids, video_dir, sample_stride=4, clip_frames=16,
                 clip_stride=4):
        self.video_dir = video_dir
        self.clip_frames = clip_frames
        self.clip_stride = clip_stride

        # Build list of (video_id, center_frame, n_frames) tuples
        self.samples = []
        self.video_id_to_indices = {}

        for vid in video_ids:
            vpath = os.path.join(video_dir, f"{vid}.mp4")
            if not os.path.exists(vpath):
                continue
            n_frames, fps = get_video_info(vpath)
            start_idx = len(self.samples)
            for frame_idx in range(0, n_frames, sample_stride):
                self.samples.append((vid, frame_idx, n_frames))
            end_idx = len(self.samples)
            self.video_id_to_indices[vid] = (start_idx, end_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, center, n_frames = self.samples[idx]
        video_path = os.path.join(self.video_dir, f"{vid}.mp4")
        try:
            clip = load_clip(video_path, center,
                             num_frames=self.clip_frames, stride=self.clip_stride,
                             n_total=n_frames)
        except Exception:
            clip = torch.zeros(3, self.clip_frames, CROP_SIZE, CROP_SIZE)
        return clip, vid


# ---------------------------------------------------------------------------
# Build frame list with weights for WeightedRandomSampler
# ---------------------------------------------------------------------------

def build_frame_list(annotations, video_dir, fps=30.0, old_labels=False):
    """Build per-frame entries for all training videos.

    Returns:
        frame_list: list of dicts {video_id, frame_idx, label, weight, n_frames}
        video_meta: dict mapping video_id -> (n_frames, fps)
    """
    frame_list = []
    video_meta = {}

    for rec in annotations:
        vid = rec["id"]
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(vpath):
            continue

        n_frames, vid_fps = get_video_info(vpath)
        if n_frames == 0:
            continue
        actual_fps = vid_fps if vid_fps > 0 else fps
        video_meta[vid] = (n_frames, actual_fps)

        target = rec["target"]
        toe = rec["time_of_event"]
        toa = rec["time_of_alert"]

        if target == 1 and toe is not None:
            event_frame = int(toe * actual_fps)
            alert_frame = int(toa * actual_fps) if toa is not None else event_frame

            # Remove last 0.25s before impact if alert duration >= 0.5s
            alert_duration = (toe - toa) if toa is not None else 0.0
            if alert_duration >= 0.5:
                cutoff_frame = int((toe - 0.25) * actual_fps)
            else:
                cutoff_frame = event_frame

            # Max valid frame: cutoff_frame (exclusive of post-crash)
            max_frame = min(cutoff_frame, n_frames - 1)

            for f in range(0, max_frame + 1):
                if alert_frame <= f <= cutoff_frame:
                    w = 30.0  # alert-to-event zone — collision imminent
                    lbl = 1
                else:
                    w = 2.0   # pre-alert frames — normal driving, sample more
                    lbl = 1 if old_labels else 0
                frame_list.append({
                    "video_id": vid,
                    "frame_idx": f,
                    "label": lbl,
                    "weight": w,
                    "n_frames": n_frames,
                })
        else:
            # Negative video: all frames, weight 1
            for f in range(n_frames):
                frame_list.append({
                    "video_id": vid,
                    "frame_idx": f,
                    "label": 0,
                    "weight": 1.0,
                    "n_frames": n_frames,
                })

    return frame_list, video_meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(pretrained=True):
    if pretrained:
        weights = MViT_V2_S_Weights.KINETICS400_V1
        model = mvit_v2_s(weights=weights)
    else:
        model = mvit_v2_s(weights=None)

    # Replace classification head for binary
    in_features = model.head[1].in_features
    model.head[1] = nn.Linear(in_features, 1)
    return model


# ---------------------------------------------------------------------------
# Cosine schedule with warmup
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                     steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_videos(model, video_ids, labels_dict, video_dir, device,
                    batch_size=8, sample_stride=4, num_workers=2):
    """Evaluate model on videos, return mAP.

    For each video, sample every sample_stride-th frame, predict, take max.
    """
    model.eval()

    eval_dataset = VideoEvalDataset(
        video_ids=video_ids,
        video_dir=video_dir,
        sample_stride=sample_stride,
        clip_frames=16,
        clip_stride=4,
    )

    if len(eval_dataset) == 0:
        return 0.0

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # Collect predictions per video
    video_preds = {}
    n_total = len(loader)
    t_eval = time.time()
    for batch_idx, (clips, vids) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = model(clips).squeeze(-1)  # (B,)
        probs = torch.sigmoid(logits).cpu().numpy()

        for prob, vid in zip(probs, vids):
            if vid not in video_preds:
                video_preds[vid] = []
            video_preds[vid].append(float(prob))

        if (batch_idx + 1) % 100 == 0:
            print(f"    eval batch {batch_idx+1}/{n_total} "
                  f"({time.time()-t_eval:.0f}s)", flush=True)

    # Aggregate: max per video
    y_true = []
    y_score = []
    for vid in video_preds:
        if vid in labels_dict:
            y_true.append(labels_dict[vid])
            y_score.append(max(video_preds[vid]))

    if len(y_true) == 0 or sum(y_true) == 0:
        return 0.0

    return average_precision_score(y_true, y_score)


# ---------------------------------------------------------------------------
# Iterative Data Cleaning
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_train_frames(model, frame_list, video_dir, device, batch_size=64,
                         num_workers=4, stride=16):
    """Run inference on training frames (every stride-th), return (index, prob) pairs."""
    model.eval()

    indices = list(range(0, len(frame_list), stride))
    sub_list = [frame_list[i] for i in indices]

    dataset = FrameLevelDataset(
        frame_list=sub_list,
        video_dir=video_dir,
        num_frames=16, stride=4,
        augment=False, augment_color=False,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    predictions = []
    n_total = len(loader)
    t0 = time.time()

    for batch_idx, (clips, _labels) in enumerate(loader):
        clips = clips.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(clips).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        predictions.extend(probs.tolist())

        if (batch_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    predict batch {batch_idx+1}/{n_total} ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Prediction complete: {len(predictions)} frames in {elapsed:.0f}s",
          flush=True)
    return list(zip(indices, predictions))


def clean_frame_list(frame_list, pred_pairs, fp_threshold=0.95, fn_threshold=0.0,
                     neighborhood=16):
    """Remove noisy frames based on model predictions.

    For each flagged frame, also removes neighboring frames from the same video
    within ±neighborhood indices in the frame list.
    """
    remove_set = set()
    n_fp = 0
    n_fn = 0

    for idx, pred in pred_pairs:
        frame = frame_list[idx]
        flagged = False

        if frame["label"] == 0 and pred > fp_threshold:
            n_fp += 1
            flagged = True
        elif fn_threshold > 0 and frame["label"] == 1 and pred < fn_threshold:
            n_fn += 1
            flagged = True

        if flagged:
            vid = frame["video_id"]
            lo = max(0, idx - neighborhood)
            hi = min(len(frame_list), idx + neighborhood + 1)
            for j in range(lo, hi):
                if frame_list[j]["video_id"] == vid:
                    remove_set.add(j)

    cleaned = [f for i, f in enumerate(frame_list) if i not in remove_set]

    print(f"  Flagged: {n_fp} false positives (label=0, pred>{fp_threshold})"
          + (f", {n_fn} ambiguous positives (label=1, pred<{fn_threshold})"
             if fn_threshold > 0 else ""))
    print(f"  Total removed (incl. neighbors): {len(remove_set)}")
    print(f"  Frames: {len(frame_list)} -> {len(cleaned)} "
          f"({100 * len(remove_set) / max(1, len(frame_list)):.1f}% removed)")

    return cleaned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Nexar E2E Frame-Level Training (Recipe B)")
    parser.add_argument("--results_dir", type=str,
                        default="results")
    parser.add_argument("--idea_id", type=str, default="e2e_mvit_v2_s")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--samples_per_epoch", type=int, default=6000)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--eval_sample_stride", type=int, default=60,
                        help="Stride for frame sampling during eval (60=~0.5fps)")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Evaluate every N epochs (always eval epoch 1)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (BADAS used 5.0)")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for AdamW")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="Center crop size (must be 224 for MViTv2-S)")
    parser.add_argument("--freeze_blocks", type=int, default=0,
                        help="Freeze first N MViT blocks + conv_proj + pos_encoding (0=no freeze)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for BCE loss (e.g., 0.05)")
    parser.add_argument("--augment_color", action="store_true",
                        help="Add ColorJitter + random grayscale augmentation")
    parser.add_argument("--cleaning_rounds", type=int, default=0,
                        help="Iterative data cleaning rounds (0=disabled)")
    parser.add_argument("--clean_epochs", type=int, default=3,
                        help="Epochs per cleaning round before predicting")
    parser.add_argument("--clean_fp_threshold", type=float, default=0.95,
                        help="Remove label=0 frames with pred > threshold")
    parser.add_argument("--clean_fn_threshold", type=float, default=0.0,
                        help="Remove label=1 frames with pred < threshold (0=disabled)")
    parser.add_argument("--clean_pred_stride", type=int, default=16,
                        help="Predict every Nth frame during cleaning (16=fast, 1=thorough)")
    parser.add_argument("--old_labels", action="store_true",
                        help="Old label mode: ALL frames in positive videos are label=1 (not just alert-to-event)")
    parser.add_argument("--save_every_eval", action="store_true",
                        help="Save checkpoint at every eval epoch (for model soup)")
    parser.add_argument("--ffr_lambda", type=float, default=0.0,
                        help="Future-Frame Regularization loss weight (RiskProp). 0=disabled.")
    parser.add_argument("--exclude_videos", type=str, default=None,
                        help="Path to file listing video IDs to exclude (one per line)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Override global crop size if specified
    global CROP_SIZE
    CROP_SIZE = args.crop_size

    set_seed(args.seed)

    # Output directory
    out_dir = Path(args.results_dir) / args.idea_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU will be extremely slow.")

    # ------------------------------------------------------------------
    # Load annotations
    # ------------------------------------------------------------------
    print("Loading annotations...")
    train_annots = load_train_annotations(TRAIN_CSV)
    test_annots = load_test_annotations(TEST_CSV)

    test_labels = {r["id"]: r["target"] for r in test_annots}
    test_video_ids = list(test_labels.keys())

    # Exclude videos from embedding-space curation
    if args.exclude_videos and os.path.exists(args.exclude_videos):
        with open(args.exclude_videos) as f:
            exclude_set = {line.strip() for line in f if line.strip()}
        before = len(train_annots)
        train_annots = [r for r in train_annots if r["id"] not in exclude_set]
        n_excluded = before - len(train_annots)
        print(f"Excluded {n_excluded} videos from {args.exclude_videos} ({before} -> {len(train_annots)})")

    # Use all training data (no val split — val mAP is uncorrelated with test)
    random.shuffle(train_annots)
    train_annots_split = train_annots

    print(f"Train videos: {len(train_annots_split)}")
    print(f"Test videos: {len(test_video_ids)}")

    # ------------------------------------------------------------------
    # Build frame list for training
    # ------------------------------------------------------------------
    print("Building frame list (scanning videos)...")
    t0 = time.time()
    frame_list, video_meta = build_frame_list(train_annots_split, TRAIN_VIDEO_DIR,
                                               old_labels=args.old_labels)
    weights = [f["weight"] for f in frame_list]
    print(f"Total frames indexed: {len(frame_list)} ({time.time()-t0:.1f}s)")

    n_pos = sum(1 for f in frame_list if f["label"] == 1)
    n_neg = len(frame_list) - n_pos
    print(f"  Positive frames: {n_pos}, Negative frames: {n_neg}")

    # ------------------------------------------------------------------
    # Iterative Data Cleaning (runs before main training)
    # ------------------------------------------------------------------
    if args.cleaning_rounds > 0:
        print(f"\n{'='*80}")
        print(f"ITERATIVE DATA CLEANING: {args.cleaning_rounds} rounds, "
              f"{args.clean_epochs} epochs each, pred_stride={args.clean_pred_stride}")
        print(f"Thresholds: FP>{args.clean_fp_threshold}, FN<{args.clean_fn_threshold}")
        print(f"{'='*80}\n")

        for clean_round in range(1, args.cleaning_rounds + 1):
            print(f"\n--- Cleaning Round {clean_round}/{args.cleaning_rounds} ---")
            print(f"Training {args.clean_epochs} epochs on {len(frame_list)} frames...")

            clean_model = build_model(pretrained=True).to(device)
            clean_params = list(clean_model.parameters())
            clean_opt = torch.optim.AdamW(
                clean_params, lr=args.lr, weight_decay=args.weight_decay)
            clean_sched = get_cosine_schedule_with_warmup(
                clean_opt, 1, args.clean_epochs,
                args.samples_per_epoch // args.batch_size)
            clean_scaler = torch.amp.GradScaler("cuda")
            clean_criterion = nn.BCEWithLogitsLoss()

            clean_ds = FrameLevelDataset(
                frame_list=frame_list, video_dir=TRAIN_VIDEO_DIR,
                num_frames=16, stride=4, augment=True,
                augment_color=args.augment_color,
            )

            for clean_ep in range(1, args.clean_epochs + 1):
                clean_model.train()
                sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=args.samples_per_epoch,
                    replacement=True,
                )
                loader = DataLoader(
                    clean_ds, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=False,
                )

                running_loss = 0.0
                n_batches = 0
                ep_start = time.time()

                for clips, labels in loader:
                    clips = clips.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    clean_opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda"):
                        logits = clean_model(clips).squeeze(-1)
                        loss = clean_criterion(logits, labels)
                    clean_scaler.scale(loss).backward()
                    clean_scaler.unscale_(clean_opt)
                    torch.nn.utils.clip_grad_norm_(clean_params, 1.0)
                    clean_scaler.step(clean_opt)
                    clean_scaler.update()
                    clean_sched.step()
                    running_loss += loss.item()
                    n_batches += 1

                del loader, sampler
                gc.collect()
                print(f"  Clean ep {clean_ep}/{args.clean_epochs} | "
                      f"loss={running_loss/max(n_batches,1):.4f} | "
                      f"time={time.time()-ep_start:.0f}s", flush=True)

            # Predict on training frames
            print(f"\nPredicting on training frames (stride={args.clean_pred_stride})...")
            pred_pairs = predict_train_frames(
                clean_model, frame_list, TRAIN_VIDEO_DIR, device,
                batch_size=args.eval_batch_size * 2,
                num_workers=args.num_workers,
                stride=args.clean_pred_stride,
            )

            # Clean
            frame_list = clean_frame_list(
                frame_list, pred_pairs,
                fp_threshold=args.clean_fp_threshold,
                fn_threshold=args.clean_fn_threshold,
                neighborhood=args.clean_pred_stride,
            )
            weights = [f["weight"] for f in frame_list]

            n_pos = sum(1 for f in frame_list if f["label"] == 1)
            n_neg = len(frame_list) - n_pos
            print(f"  After round {clean_round}: {n_pos} pos, {n_neg} neg frames")

            del clean_model, clean_opt, clean_sched, clean_scaler
            del clean_criterion, clean_ds, pred_pairs
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\n{'='*80}")
        print(f"Cleaning complete. Final: {len(frame_list)} frames "
              f"({n_pos} pos, {n_neg} neg)")
        print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("Building MViTv2-S model...")
    model = build_model(pretrained=True)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Layer freezing (anti-overfitting for small datasets)
    # ------------------------------------------------------------------
    if args.freeze_blocks > 0:
        # Freeze conv_proj and pos_encoding
        for param in model.conv_proj.parameters():
            param.requires_grad = False
        for param in model.pos_encoding.parameters():
            param.requires_grad = False
        # Freeze first N blocks
        for i in range(min(args.freeze_blocks, len(model.blocks))):
            for param in model.blocks[i].parameters():
                param.requires_grad = False
        n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"Frozen: {n_frozen} params, Trainable: {n_trainable} params "
              f"({sum(p.numel() for p in model.parameters() if p.requires_grad):,} weights)")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.samples_per_epoch // args.batch_size
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, steps_per_epoch
    )
    scaler = torch.amp.GradScaler("cuda")
    if args.label_smoothing > 0:
        # Smooth labels: 0 -> eps, 1 -> 1-eps
        ls = args.label_smoothing
        print(f"Label smoothing: {ls} (targets: [{ls:.3f}, {1-ls:.3f}])")
    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 1
    best_test_map = 0.0
    best_epoch = 0
    no_improve = 0

    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            # Advance scheduler to match checkpoint epoch
            for _ in range(ckpt["epoch"] * steps_per_epoch):
                scheduler.step()
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_test_map = ckpt.get("best_test_map", ckpt.get("test_map", 0.0))
        best_epoch = ckpt["epoch"]
        print(f"  Resumed at epoch {start_epoch}, best_test_map={best_test_map:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_start = time.time()

    train_dataset = FrameLevelDataset(
        frame_list=frame_list,
        video_dir=TRAIN_VIDEO_DIR,
        num_frames=16,
        stride=4,
        augment=True,
        augment_color=args.augment_color,
        ffr=args.ffr_lambda > 0,
    )

    print(f"\n{'='*80}")
    print(f"Starting training: {args.epochs} epochs, {args.samples_per_epoch} samples/epoch")
    if args.ffr_lambda > 0:
        print(f"  RiskProp FFR enabled: lambda={args.ffr_lambda}")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Create DataLoader fresh each epoch to avoid semaphore leaks
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=args.samples_per_epoch,
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
        )

        # ---- Train ----
        model.train()

        running_loss = 0.0
        n_batches = 0
        total_batches = args.samples_per_epoch // args.batch_size

        use_ffr = args.ffr_lambda > 0

        for batch in train_loader:
            if use_ffr:
                clips_t, clips_next, labels = batch
                clips_t = clips_t.to(device, non_blocking=True)
                clips_next = clips_next.to(device, non_blocking=True)
            else:
                clips_t, labels = batch
                clips_t = clips_t.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.label_smoothing > 0:
                labels = labels * (1 - args.label_smoothing) + args.label_smoothing * 0.5

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(clips_t).squeeze(-1)  # (B,)
                loss = criterion(logits, labels)

                if use_ffr:
                    with torch.no_grad():
                        logits_next = model(clips_next).squeeze(-1)  # (B,)
                    ffr_loss = F.mse_loss(logits, logits_next.detach())
                    loss = loss + args.ffr_lambda * ffr_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                elapsed = time.time() - epoch_start
                print(f"  batch {n_batches}/{total_batches} | "
                      f"loss={running_loss/n_batches:.4f} | "
                      f"elapsed={elapsed:.0f}s", flush=True)

        avg_loss = running_loss / max(n_batches, 1)

        # Clean up DataLoader workers before eval to free resources
        del train_loader
        del sampler
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Eval (test only, skip val since val mAP is uncorrelated) ----
        do_eval = (epoch == 1 or epoch % args.eval_every == 0
                   or epoch == args.epochs)
        if do_eval:
            test_map = evaluate_videos(
                model, test_video_ids, test_labels, TEST_VIDEO_DIR, device,
                batch_size=args.eval_batch_size,
                sample_stride=args.eval_sample_stride,
            )
        else:
            test_map = -1.0

        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        if do_eval:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"loss={avg_loss:.4f} | test_mAP={test_map:.4f} | "
                  f"lr={lr_now:.2e} | time={epoch_time:.1f}s", flush=True)
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"loss={avg_loss:.4f} | (no eval) | "
                  f"lr={lr_now:.2e} | time={epoch_time:.1f}s", flush=True)

        # Track best by test mAP (since we have labels)
        if do_eval and test_map > best_test_map:
            best_test_map = test_map
            best_epoch = epoch
            no_improve = 0

            # Save checkpoint
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "test_map": test_map,
                "best_test_map": test_map,
            }, ckpt_path)
            print(f"  -> New best! Saved to {ckpt_path}", flush=True)
        elif do_eval:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

        if do_eval and args.save_every_eval:
            ep_ckpt = out_dir / f"checkpoint_ep{epoch}.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "test_map": test_map}, ep_ckpt)

    total_time = time.time() - total_start

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    metrics = {
        "map": best_test_map,
        "test_map": best_test_map,
        "backbone": "mvit_v2_s",
        "model_type": "e2e_frame_level",
        "training_time": round(total_time, 1),
        "epoch": epoch,
        "best_epoch": best_epoch,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best test mAP: {best_test_map:.4f} (epoch {best_epoch})")
    print(f"Metrics saved to {metrics_path}")
    print(f"Model saved to  {out_dir / 'best_model.pt'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
