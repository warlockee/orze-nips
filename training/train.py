#!/usr/bin/env python3
"""Nexar Collision Detection — train.py for orze orchestrator.

Loads pre-extracted vision backbone features (.pt), trains a temporal
classifier, evaluates on held-out val split and test set, writes
metrics.json for orze.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def deep_merge(base, override):
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(args):
    base_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    idea_cfg_path = Path(args.results_dir) / args.idea_id / "idea_config.yaml"
    idea_cfg = {}
    if idea_cfg_path.exists():
        idea_cfg = yaml.safe_load(idea_cfg_path.read_text()) or {}
    return deep_merge(base_cfg, idea_cfg)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FeatureDataset(Dataset):
    def __init__(self, file_list, max_seq_len, augment=False, aug_cfg=None,
                 truncation="end", remove_post_crash=False, post_crash_margin=1,
                 feature_norm=False, alert_crop=False, alert_crop_padding=5):
        self.file_list = file_list
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        self.truncation = truncation  # "end" = keep last frames, "uniform" = downsample, "start" = keep first
        self.remove_post_crash = remove_post_crash
        self.post_crash_margin = post_crash_margin  # frames to keep after event (1=include event frame)
        self.feature_norm = feature_norm  # L2-normalize per-frame features
        self.alert_crop = alert_crop  # crop positive samples to alert-to-event window
        self.alert_crop_padding = alert_crop_padding  # frames before alert to include

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rec = self.file_list[idx]
        feats = rec["features"]  # (T, D)
        label = rec["label"]
        T, D = feats.shape

        if self.feature_norm:
            feats = F.normalize(feats, p=2, dim=-1)

        event_frame = rec.get("event_frame")

        # Data curation: remove post-crash frames (1st place technique).
        # Applied to BOTH train and val so end-truncation captures pre-crash frames.
        if self.remove_post_crash and event_frame is not None and event_frame < T:
            end_idx = min(T, event_frame + self.post_crash_margin)
            feats = feats[:end_idx]
            T = feats.shape[0]

        # Alert-window crop: for positive samples, keep only [alert-padding, event+margin].
        # Removes normal-driving frames that dilute the collision signal.
        # Applied to train only — test has no alert timestamps.
        alert_frame = rec.get("alert_frame")
        if self.alert_crop and alert_frame is not None and event_frame is not None:
            start_a = max(0, alert_frame - self.alert_crop_padding)
            end_a = min(T, event_frame + self.post_crash_margin)
            if end_a > start_a:
                feats = feats[start_a:end_a]
                T = feats.shape[0]

        # Event-aware temporal crop: center window on collision event with jitter
        event_cropped = False
        if self.augment and self.aug_cfg.get("event_crop", False) and event_frame is not None:
            crop_len = max(1, int(T * self.aug_cfg.get("event_crop_ratio", 0.5)))
            jitter = int(self.aug_cfg.get("event_crop_jitter", 0.1) * T)
            center = event_frame + random.randint(-jitter, jitter)
            start = max(0, min(center - crop_len // 2, T - crop_len))
            feats = feats[start : start + crop_len]
            T = feats.shape[0]
            event_cropped = True
        # Standard temporal crop augmentation (fallback for negatives / no timestamp)
        elif self.augment and self.aug_cfg.get("temporal_crop", False):
            ratio = self.aug_cfg.get("temporal_crop_ratio", 0.8)
            crop_len = max(1, int(T * ratio))
            if self.aug_cfg.get("temporal_crop_end", False):
                # Bias crop toward the end (collision region)
                min_start = max(0, T - crop_len - int(T * 0.2))
                start = random.randint(min_start, T - crop_len)
            else:
                start = random.randint(0, T - crop_len)
            feats = feats[start : start + crop_len]
            T = feats.shape[0]

        # Pad or truncate to max_seq_len
        if T > self.max_seq_len:
            if event_cropped:
                # After event_crop, event is at center — keep center frames
                mid = T // 2
                half = self.max_seq_len // 2
                start_t = max(0, min(mid - half, T - self.max_seq_len))
                feats = feats[start_t : start_t + self.max_seq_len]
            elif self.truncation == "end":
                feats = feats[-self.max_seq_len:]
            elif self.truncation == "start":
                feats = feats[:self.max_seq_len]
            else:
                indices = torch.linspace(0, T - 1, self.max_seq_len).long()
                feats = feats[indices]
            mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        elif T < self.max_seq_len:
            pad = torch.zeros(self.max_seq_len - T, D)
            feats = torch.cat([feats, pad], dim=0)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            mask[:T] = True
        else:
            mask = torch.ones(self.max_seq_len, dtype=torch.bool)

        # Feature dropout augmentation
        if self.augment and self.aug_cfg.get("feature_dropout", 0) > 0:
            drop_mask = torch.bernoulli(
                torch.full((self.max_seq_len, 1), 1 - self.aug_cfg["feature_dropout"])
            )
            feats = feats * drop_mask

        return feats, mask, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AttentiveProbe(nn.Module):
    """Learned multi-query attention pooling (inspired by BADAS attentive probe).
    M learned queries attend to the temporal sequence, producing M×d features
    that are concatenated and projected to a single vector."""
    def __init__(self, hidden_dim, num_queries=8, query_dim=None, dropout=0.3):
        super().__init__()
        query_dim = query_dim or hidden_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(num_queries * hidden_dim, hidden_dim)

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T) True=valid
        B, T, D = x.shape
        Q = self.queries.expand(B, -1, -1)  # (B, M, D)
        # Scaled dot-product attention: Q @ K^T / sqrt(D)
        scores = torch.bmm(Q, x.transpose(1, 2)) / (D ** 0.5)  # (B, M, T)
        # Mask out padding positions
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        attn = torch.softmax(scores, dim=-1)  # (B, M, T)
        attn = self.attn_dropout(attn)
        # Weighted sum: (B, M, T) @ (B, T, D) -> (B, M, D)
        out = torch.bmm(attn, x)  # (B, M, D)
        out = out.reshape(B, -1)  # (B, M*D)
        return self.out_proj(out)  # (B, D)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=4,
                 dropout=0.3, pool="cls", num_queries=8):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=512)
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        if pool == "attention":
            self.attn_pool = AttentiveProbe(hidden_dim, num_queries=num_queries, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T) True=valid
        x = self.proj(x)
        x = self.pos_enc(x)

        if self.pool == "cls":
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.ones(B, 1, device=mask.device, dtype=torch.bool)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer expects src_key_padding_mask: True=ignored
        padding_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        if self.pool == "cls":
            out = x[:, 0]
        elif self.pool == "mean":
            # Masked mean pooling
            mask_f = mask.unsqueeze(-1).float()
            out = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        elif self.pool == "max":
            x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            out = x.max(dim=1).values
        elif self.pool == "attention":
            out = self.attn_pool(x, mask)
        else:
            out = x[:, 0]

        return self.head(out).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        # h: (num_layers*2, B, hidden_dim) — take last layer fwd+bwd
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(h).squeeze(-1)


class FusionClassifier(nn.Module):
    """Late fusion: independent per-backbone encoders + fusion head."""
    def __init__(self, backbone_dims, hidden_dim=256, num_layers=4, num_heads=4,
                 dropout=0.3, pool="cls"):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for name, dim in backbone_dims.items():
            self.encoders[name] = TransformerClassifier(
                dim, hidden_dim, num_layers, num_heads, dropout, pool
            )
            # Remove the classification head — we'll use a shared one
            self.encoders[name].head = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * len(backbone_dims), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features_dict, masks_dict):
        embeddings = []
        for name, encoder in self.encoders.items():
            embeddings.append(encoder(features_dict[name], masks_dict[name]))
        fused = torch.cat(embeddings, dim=-1)
        return self.head(fused).squeeze(-1)


def build_model(cfg, input_dim):
    mcfg = cfg["model"]
    model_type = mcfg.get("type", "transformer")
    kwargs = {
        "input_dim": input_dim,
        "hidden_dim": mcfg.get("hidden_dim", 256),
        "num_layers": mcfg.get("num_layers", 4),
        "dropout": mcfg.get("dropout", 0.3),
    }
    if model_type == "transformer":
        kwargs["num_heads"] = mcfg.get("num_heads", 4)
        kwargs["pool"] = mcfg.get("pool", "cls")
        kwargs["num_queries"] = mcfg.get("num_queries", 8)
        return TransformerClassifier(**kwargs)
    elif model_type == "gru":
        return GRUClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

BACKBONE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "siglip2_vit_b16": 768,
    "vjepa2_vitl": 1024,
}


def load_timestamps(csv_path):
    """Load time_of_event and time_of_alert from train.csv, keyed by video id."""
    ts_map = {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return ts_map
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["id"]
            toe = row.get("time_of_event", "").strip()
            toa = row.get("time_of_alert", "").strip()
            ts_map[vid] = {
                "time_of_event": float(toe) if toe else None,
                "time_of_alert": float(toa) if toa else None,
            }
    return ts_map


def load_split(feature_dir, backbone, max_seq_len=256, timestamps=None):
    d = Path(feature_dir) / backbone
    records = []
    for pt_file in sorted(d.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        vid = data.get("video_id", pt_file.stem)
        rec = {
            "features": data["features"],
            "label": data["label"],
            "video_id": vid,
        }
        fps = data.get("fps", 5.0)
        if timestamps and vid in timestamps:
            ts = timestamps[vid]
            if ts["time_of_event"] is not None:
                rec["event_frame"] = int(ts["time_of_event"] * fps)
            if ts["time_of_alert"] is not None:
                rec["alert_frame"] = int(ts["time_of_alert"] * fps)
        records.append(rec)
    return records


def load_multi_backbone(feature_dir, backbones, max_seq_len=256):
    """Load features from multiple backbones, aligned by video_id."""
    all_data = {}
    for bb in backbones:
        for rec in load_split(feature_dir, bb, max_seq_len):
            vid = rec["video_id"]
            if vid not in all_data:
                all_data[vid] = {"label": rec["label"], "video_id": vid, "backbones": {}}
            all_data[vid]["backbones"][bb] = rec["features"]
    # Only keep videos that have ALL backbones
    complete = {
        vid: rec for vid, rec in all_data.items()
        if len(rec["backbones"]) == len(backbones)
    }
    return list(complete.values())


def train_val_split(records, val_ratio=0.15, seed=42, balanced_val=False):
    rng = random.Random(seed)
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if balanced_val:
        # Balanced val: equal pos/neg to match test distribution (50/50)
        n_val_pos = int(len(pos) * val_ratio)
        n_val_neg = n_val_pos  # match positive count for balance
        n_val_neg = min(n_val_neg, len(neg))
    else:
        # Stratified: maintain class proportions
        n_val_pos = int(len(pos) * val_ratio)
        n_val_neg = int(len(neg) * val_ratio)
    val = pos[:n_val_pos] + neg[:n_val_neg]
    train = pos[n_val_pos:] + neg[n_val_neg:]
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for feats, mask, labels in loader:
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(feats, mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for feats, mask, labels in loader:
        feats, mask = feats.to(device), mask.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(feats, mask)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    ap = average_precision_score(all_labels, all_probs)
    return ap, all_probs, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--ideas-md", required=True)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()

    cfg = load_config(args)
    idea_dir = Path(args.results_dir) / args.idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("eval", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}, idea={args.idea_id}")
    print(f"[train] config: {json.dumps(cfg, indent=2, default=str)}")

    t0 = time.time()

    backbone = cfg.get("backbone", "dinov3_vitl16")
    fusion_backbones = cfg.get("fusion_backbones", [])
    max_seq_len = cfg.get("max_seq_len", 256)
    feature_dir = cfg.get("feature_dir", "features/nexar")
    test_feature_dir = cfg.get("test_feature_dir", "features/nexar_test")
    tcfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})
    eval_cfg = cfg.get("eval", {})

    # --- Load timestamps from train.csv ---
    train_csv = cfg.get("train_csv", os.environ.get("NEXAR_TRAIN_CSV", "data/train.csv"))
    timestamps = load_timestamps(train_csv)
    if timestamps:
        n_with_event = sum(1 for v in timestamps.values() if v["time_of_event"] is not None)
        print(f"[train] Loaded timestamps for {len(timestamps)} videos ({n_with_event} with event times)")

    # --- Load data ---
    if fusion_backbones:
        raise NotImplementedError("Fusion mode not yet wired — use single backbone")
    else:
        input_dim = BACKBONE_DIMS.get(backbone)
        if input_dim is None:
            raise ValueError(f"Unknown backbone: {backbone}")

        all_train = load_split(feature_dir, backbone, max_seq_len, timestamps=timestamps)
        print(f"[train] Loaded {len(all_train)} train videos, backbone={backbone}, dim={input_dim}")

        train_records, val_records = train_val_split(
            all_train, val_ratio=eval_cfg.get("val_ratio", 0.15), seed=seed,
            balanced_val=eval_cfg.get("balanced_val", False),
        )

        # Check if backbone exists in test set
        test_backbone_dir = Path(test_feature_dir) / backbone
        if test_backbone_dir.exists():
            test_records = load_split(test_feature_dir, backbone, max_seq_len)
            print(f"[train] Loaded {len(test_records)} test videos")
        else:
            print(f"[train] WARN: backbone {backbone} not in test set, using val only")
            test_records = None

    truncation = cfg.get("truncation", "end")
    data_cfg = cfg.get("data_curation", {})
    remove_post_crash = data_cfg.get("remove_post_crash", False)
    post_crash_margin = data_cfg.get("post_crash_margin", 1)
    feature_norm = data_cfg.get("feature_norm", False)
    alert_crop = data_cfg.get("alert_crop", False)
    alert_crop_padding = data_cfg.get("alert_crop_padding", 5)
    if remove_post_crash:
        n_trimmed = sum(1 for r in train_records if r.get("event_frame") is not None)
        print(f"[train] Post-crash removal enabled: {n_trimmed} positive videos will be trimmed (margin={post_crash_margin})")
    if feature_norm:
        print(f"[train] L2 feature normalization enabled")
    if alert_crop:
        n_with_alert = sum(1 for r in train_records if r.get("alert_frame") is not None)
        print(f"[train] Alert-window crop enabled: {n_with_alert} positive videos will be cropped to alert window (padding={alert_crop_padding})")

    train_ds = FeatureDataset(train_records, max_seq_len, augment=True, aug_cfg=aug_cfg,
                              truncation=truncation, remove_post_crash=remove_post_crash,
                              post_crash_margin=post_crash_margin, feature_norm=feature_norm,
                              alert_crop=alert_crop, alert_crop_padding=alert_crop_padding)
    val_ds = FeatureDataset(val_records, max_seq_len, augment=False, truncation=truncation,
                            remove_post_crash=remove_post_crash,
                            post_crash_margin=post_crash_margin, feature_norm=feature_norm)

    # Sample weighting via WeightedRandomSampler (1st place technique)
    sw_cfg = data_cfg.get("sample_weights", {})
    batch_size = tcfg.get("batch_size", 64)
    if sw_cfg.get("enabled", False):
        pos_w = sw_cfg.get("positive_weight", 3.0)
        neg_w = sw_cfg.get("negative_weight", 1.0)
        weights = [pos_w if r["label"] == 1 else neg_w for r in train_records]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"[train] WeightedRandomSampler: pos_weight={pos_w}, neg_weight={neg_w}")
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.get("batch_size", 64) * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    if test_records:
        test_ds = FeatureDataset(test_records, max_seq_len, augment=False, truncation=truncation,
                                 feature_norm=feature_norm)
        test_loader = DataLoader(
            test_ds, batch_size=tcfg.get("batch_size", 64) * 2, shuffle=False,
            num_workers=4, pin_memory=True,
        )
    else:
        test_loader = None

    # --- Build model ---
    model = build_model(cfg, input_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model params: {param_count:,}")

    # --- Optimizer & scheduler ---
    opt_name = tcfg.get("optimizer", "adamw")
    lr = tcfg.get("lr", 3e-4)
    wd = tcfg.get("weight_decay", 1e-2)

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    epochs = tcfg.get("epochs", 60)
    warmup_epochs = tcfg.get("warmup_epochs", 5)

    scheduler_type = tcfg.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None

    # --- Loss ---
    pw = tcfg.get("pos_weight", 2.8)
    ls = tcfg.get("label_smoothing", 0.05)
    loss_type = tcfg.get("loss", "bce")
    focal_gamma = tcfg.get("focal_gamma", 2.0)
    focal_alpha = tcfg.get("focal_alpha", 0.25)

    if loss_type == "focal":
        def criterion(logits, labels):
            if ls > 0:
                labels = labels * (1 - ls) + (1 - labels) * ls
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * labels + (1 - p) * (1 - labels)
            alpha_t = focal_alpha * labels + (1 - focal_alpha) * (1 - labels)
            focal_weight = alpha_t * (1 - p_t) ** focal_gamma
            return (focal_weight * bce).mean()
        print(f"[train] Focal loss: gamma={focal_gamma}, alpha={focal_alpha}")
    else:
        pos_weight = torch.tensor([pw], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if ls > 0:
            _base_criterion = criterion
            def criterion(logits, labels):
                labels_smooth = labels * (1 - ls) + (1 - labels) * ls
                return _base_criterion(logits, labels_smooth)

    scaler = torch.amp.GradScaler("cuda")

    # --- Train ---
    best_val_ap = 0
    best_epoch = 0
    patience = tcfg.get("patience", 20)
    no_improve = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_ap, _, _ = evaluate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1:3d}/{epochs}] loss={train_loss:.4f} val_mAP={val_ap:.4f} lr={cur_lr:.2e}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), idea_dir / "best_model.pt")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[train] Early stopping at epoch {epoch+1}, best={best_val_ap:.4f} at epoch {best_epoch}")
            break

    # --- Evaluate best model ---
    model.load_state_dict(torch.load(idea_dir / "best_model.pt", map_location=device, weights_only=True))

    val_ap, _, _ = evaluate(model, val_loader, device)
    print(f"[eval] Best val mAP = {val_ap:.4f} (epoch {best_epoch})")

    test_ap = None
    if test_loader is not None:
        test_ap, test_probs, test_labels = evaluate(model, test_loader, device)
        print(f"[eval] Test mAP = {test_ap:.4f}")

        # Save test predictions
        np.savez(
            idea_dir / "test_predictions.npz",
            probs=test_probs, labels=test_labels,
        )

    training_time = time.time() - t0

    # --- Write metrics ---
    metrics = {
        "status": "COMPLETED",
        "map": test_ap if test_ap is not None else val_ap,
        "val_map": val_ap,
        "test_map": test_ap,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "training_time": round(training_time, 1),
        "backbone": backbone,
        "model_type": cfg["model"]["type"],
        "param_count": param_count,
    }

    temp = idea_dir / "metrics.json.tmp"
    temp.write_text(json.dumps(metrics, indent=2))
    temp.replace(idea_dir / "metrics.json")

    print(f"[done] metrics written to {idea_dir / 'metrics.json'}")
    print(f"[done] mAP={metrics['map']:.4f}, time={training_time:.0f}s")
    return 0


if __name__ == "__main__":
    exit(main())
