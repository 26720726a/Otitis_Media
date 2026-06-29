import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


# ==========================================================
# 태스크 정의 (crop 방식: 한쪽 귀 crop → 2 head)
#   0: temporal area  (측두골)
#   1: otitis media   (중이염)
#
# 입력 이미지는 train/inference 쪽에서 좌/우로 잘라 전달.
# 이 모델은 "이미 한쪽만 담긴 crop"을 받아 2개 태스크 분류.
# ==========================================================
NUM_TASKS  = 2
TASK_NAMES = ["temporal", "otitis"]


def is_temporal_label(image_number_value):
    """CSV의 Image number 값으로 태스크 인덱스 반환.

    "temporal" 포함 → 0 (temporal head)
    그 외           → 1 (otitis head)
    """
    img = str(image_number_value).strip().lower()
    return 0 if "temporal" in img else 1


class MyModel(nn.Module):
    """
    2-Head ConvNeXt-Tiny (crop 입력 전용)

    구조:
        한쪽 귀 crop → ConvNeXt-Tiny backbone (공유)
                     → shared FC (태스크 간 연관성 학습)
                     → 2개 독립 head (temporal / otitis)

    기존 4-head 모델과 동일: backbone, freeze 범위(실험 B ~42%),
    shared layer(768→512→128), dropout(0.5/0.3/0.3), weight init.
    변경점: NUM_TASKS 4→2, heads 4→2, forward 출력 (B,2,2).
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # ── [1] Backbone: ConvNeXt-Tiny (ImageNet pretrained) ──
        backbone      = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_feat       = backbone.classifier[2].in_features  # 768
        self.features = backbone.features   # (B, 768, H, W) 출력
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        # ── [2] Early Layer Freezing (실험 B: ~42% frozen) ──
        # features[0]~[5]: stem + stage1 + downsample + stage2 + downsample + stage3
        freeze_layers = [
            backbone.features[0],   # stem
            backbone.features[1],   # stage1
            backbone.features[2],   # downsample
            backbone.features[3],   # stage2
            backbone.features[4],   # downsample
            backbone.features[5],   # stage3
        ]
        for layer in freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # ── [3] Shared Intermediate Layer ──
        self.shared = nn.Sequential(
            nn.LayerNorm(in_feat),
            nn.Dropout(0.5),
            nn.Linear(in_feat, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
        )

        # ── [4] Task-specific Heads (2개) ──
        self.heads = nn.ModuleList([
            nn.Linear(128, num_classes) for _ in range(NUM_TASKS)
        ])

        # ── Weight Initialization ──
        for m in list(self.shared.modules()) + list(self.heads.modules()):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) — 한쪽 귀만 담긴 crop 이미지
        Returns:
            (B, 2, 2) — (배치, 2 task, 2 class)
        """
        f = self.features(x)    # (B, 768, H, W)
        f = self.avgpool(f)     # (B, 768, 1, 1)
        f = torch.flatten(f, 1) # (B, 768)

        s = self.shared(f)      # (B, 128)

        return torch.stack([h(s) for h in self.heads], dim=1)  # (B, 2, 2)


def get_model():
    return MyModel()
