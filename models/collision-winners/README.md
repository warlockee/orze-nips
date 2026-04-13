# Collision Detection Winners

Top 3 models from 639 validated experiments. All use DINOv2 backbone + Zipformer temporal encoder.

## Results

| Rank | Model | AUC-ROC | F1 | FPR | Recall | Precision | MAE | Backbone | Params |
|------|-------|---------|----|-----|--------|-----------|-----|----------|--------|
| 1 | mustan-vitb-zipformer | 0.788 | 0.769 | 0.364 | 0.833 | 0.714 | 47.0 | dinov2_vitb14 | 125.7M |
| 2 | nexvitad-bottleneck-zipformer | 0.758 | 0.762 | 0.091 | 0.667 | 0.889 | 13.9 | dinov2_vitb14 | 104.2M |
| 3 | ttc-geometric-vitl-zipformer | 0.705 | 0.828 | 0.455 | 1.000 | 0.706 | 26.2 | dinov2_vitl14 | 348.6M |

## Model Strengths

- **mustan-vitb-zipformer**: Best overall AUC (0.788). Multi-scale temporal context forces motion-over-appearance.
- **nexvitad-bottleneck-zipformer**: Lowest false positive rate (9.1%). Bottleneck architecture compresses to domain-invariant features.
- **ttc-geometric-vitl-zipformer**: Perfect recall (1.000). Never misses a collision. Best F1 (0.828). ViT-L captures geometric/depth cues.

## Checkpoints

Checkpoints stored on FSx (not in git):
- `nexvitad-bottleneck-zipformer`: `/home/ec2-user/fsx/vlm/checkpoints/collision/idea-192/best.pt` (1.2GB)
- `ttc-geometric-vitl-zipformer`: No checkpoint saved (retrain from recipe)
- `mustan-vitb-zipformer`: No checkpoint saved (retrain from recipe)

## File Structure

```
models/collision-winners/
├── README.md
├── mustan-vitb-zipformer/
│   ├── recipe.yaml             # Full config to reproduce
│   ├── metrics.json            # Training metrics
│   └── validation_report.json  # Real-world validation results
├── nexvitad-bottleneck-zipformer/
│   ├── recipe.yaml
│   ├── metrics.json
│   └── validation_report.json
└── ttc-geometric-vitl-zipformer/
    ├── recipe.yaml
    ├── metrics.json
    └── validation_report.json
```
