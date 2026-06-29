import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


# ==========================================================
# 태스크 정의 (crop 방식: 한쪽 귀 crop → 3 head)
#   0: temporal area  (측두골)         ← 양쪽 crop 공유
#   1: rt_otitis      (우측 중이염)    ← LEFT  crop 전용 (standard 기준)
#   2: lt_otitis      (좌측 중이염)    ← RIGHT crop 전용 (standard 기준)
#
# 입력 이미지는 train/inference 쪽에서 좌/우로 잘라 전달.
# flip 없음: rt_otitis head는 LEFT crop 방향 그대로 학습,
#           lt_otitis head는 RIGHT crop 방향 그대로 학습.
# head 분리 + 라벨 라우팅으로 방향 shortcut 차단을 시도.
#
# ★ 3-head without-flip 실험 전용 격리 파일
#   train_crop_3head.py / inference_crop_3head.py 전용으로 사용.
#   체크포인트: best_model_convNeXt-Tiny_crop_3head_fold{k}.pth
#   stats:      preproc_model_convNeXt-Tiny_crop_3head_stats.json
# ==========================================================
NUM_TASKS  = 3
TASK_NAMES = ["temporal", "rt_otitis", "lt_otitis"]


def is_temporal_label(image_number_value):
    """CSV의 Image number 값으로 태스크 인덱스 반환.

    "temporal" 포함 → 0 (temporal head)
    그 외           → 1 or 2 (otitis head; rt/lt 구분은 라벨 라우팅에서 결정)
    """
    img = str(image_number_value).strip().lower()
    return 0 if "temporal" in img else 1


class MyModel(nn.Module):
    """
    3-Head ConvNeXt-Tiny (crop 입력 전용) — 3-head without-flip 실험용

    구조:
        한쪽 귀 crop → ConvNeXt-Tiny backbone (공유)
                     → shared FC (태스크 간 연관성 학습)
                     → 3개 독립 head (temporal / rt_otitis / lt_otitis)

    기존 2-head 모델과 동일: backbone, freeze 범위(features[0~5] ~43.7%),
    shared layer(768→512→128), dropout(0.5/0.3/0.3), weight init.
    변경점: NUM_TASKS 2→3, heads 2→3, forward 출력 (B,3,2).

    flip 없이 라벨 라우팅만으로 rt/lt otitis head를 분리.
    head[1]=rt_otitis는 LEFT crop(환자 Rt)만 학습,
    head[2]=lt_otitis는 RIGHT crop(환자 Lt)만 학습 (IGNORE_LABEL 마스킹).
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # ── [1] Backbone: ConvNeXt-Tiny (ImageNet pretrained) ──
        backbone      = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_feat       = backbone.classifier[2].in_features  # 768
        self.features = backbone.features   # (B, 768, H, W) 출력
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        # ── [2] Early Layer Freezing (Stage3까지, ~43.7%) ──
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

        # ── [4] Task-specific Heads (3개) ──
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
            x: (B, 3, H, W) — 한쪽 귀만 담긴 crop 이미지 (flip 없음)
        Returns:
            (B, 3, 2) — (배치, 3 task, 2 class)
        """
        f = self.features(x)    # (B, 768, H, W)
        f = self.avgpool(f)     # (B, 768, 1, 1)
        f = torch.flatten(f, 1) # (B, 768)

        s = self.shared(f)      # (B, 128)

        return torch.stack([h(s) for h in self.heads], dim=1)  # (B, 3, 2)


def get_model():
    return MyModel()
