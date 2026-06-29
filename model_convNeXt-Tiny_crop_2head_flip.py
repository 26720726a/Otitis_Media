import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


# ==========================================================
# 태스크 정의 (crop 방식: 한쪽 귀 crop → 2 head)
#   0: temporal area  (측두골)   ← 양쪽 crop 공유
#   1: otitis media   (중이염)   ← 양쪽 crop 공유 (rt/lt 공유 head)
#
# 입력 이미지는 train/inference 쪽에서 좌/우로 잘라 전달.
# temporal · otitis 모두 left/right crop이 같은 head를 공유한다.
# 3-head 대비 차이: otitis를 rt/lt로 쪼개지 않아 각 head가
# 전체 데이터를 학습한다.
#
# ★ 2-head crop-flip 증강 실험 전용 격리 파일
#   train_crop_2head_flip.py / inference_crop_2head.py 전용으로 사용.
#   체크포인트: best_model_convNeXt-Tiny_crop_2head_flip_fold{k}.pth
#   stats:      preproc_model_convNeXt-Tiny_crop_2head_flip_stats.json
#
#   아키텍처는 model_convNeXt-Tiny_crop_2head.py와 완전히 동일.
#   파일을 분리한 이유: 체크포인트·stats 이름이 기존(_2head)과 충돌하지 않게 하기 위함.
# ==========================================================
NUM_TASKS  = 2
TASK_NAMES = ["temporal", "otitis"]


def is_temporal_label(image_number_value):
    """CSV의 Image number 값으로 태스크 인덱스 반환.

    "temporal" 포함 → 0 (temporal head)
    그 외           → 1 or 2 (otitis head; rt/lt 구분은 라벨 라우팅에서 결정)
    """
    img = str(image_number_value).strip().lower()
    return 0 if "temporal" in img else 1


class MyModel(nn.Module):
    """
    2-Head ConvNeXt-Tiny (crop 입력 전용) — 2-head 공유 otitis + crop-flip 증강 실험용

    구조:
        한쪽 귀 crop → ConvNeXt-Tiny backbone (공유)
                     → shared FC (태스크 간 연관성 학습)
                     → 2개 독립 head (temporal / otitis)

    backbone, freeze 범위(features[0~5] ~43.7%),
    shared layer(768→512→128), dropout(0.5/0.3/0.3), weight init 모두 동일.
    변경점(vs 3-head): NUM_TASKS 3→2, heads 3→2, forward 출력 (B,2,2).

    otitis head(1)를 left/right crop이 함께 학습 — rt/lt 분리 없음.
    temporal head(0)과 동일한 방식으로 좌우 crop이 공유.
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
