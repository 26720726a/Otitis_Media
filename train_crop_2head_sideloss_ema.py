import os
import json
import copy
import random
import argparse
import importlib.util
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


# ==========================================================
# [모델 파일 고정]
# ==========================================================
MODEL_FILE = "model_convNeXt-Tiny_crop_2head_sideloss_ema.py"


# ==========================================================
# [EMA 헬퍼]
# ==========================================================
class ModelEMA:
    """가중치 EMA (timm 스타일, decay warmup 포함).
    update()는 raw 모델 학습에 영향 없음. 추론 땐 self.ema 사용."""
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # 짧은 학습에서 EMA가 초기 나쁜 weight에 갇히지 않도록 decay를 워밍업
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ==========================================================
# [하이퍼파라미터]
# ==========================================================
IMAGE_SIZE     = 224
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_EPOCHS     = 20
PATIENCE       = 6
NUM_FOLDS      = 4
SEED           = 42

USE_CLAHE      = True
USE_25D        = True
USE_FOCAL_LOSS = True

NUM_TASKS    = 2
TASK_NAMES   = ["temporal", "otitis"]
IGNORE_LABEL = -1


# ==========================================================
# [좌우 crop 방향 설정]
#
# "standard" : 방사선 표준 규약 (환자 우측이 이미지 왼쪽에 표시됨)
#   LEFT  crop → 환자 Rt → rt_temporal, rt_otitis 라벨
#   RIGHT crop → 환자 Lt → lt_temporal, lt_otitis 라벨
#
# "flipped"  : 반대 방향
#   LEFT  crop → 환자 Lt → lt_temporal, lt_otitis 라벨
#   RIGHT crop → 환자 Rt → rt_temporal, rt_otitis 라벨
#
# ★ 이 한 줄만 바꾸면 방향 실험 가능
# ==========================================================
CROP_SIDE_TO_RL = "standard"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# [모델 파일 동적 로드]
# ==========================================================
def load_model_class(model_file):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file}")
    if not model_file.endswith(".py"):
        raise ValueError("모델 파일은 반드시 .py 파일이어야 합니다.")

    module_name = os.path.splitext(os.path.basename(model_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"모델 파일을 import할 수 없습니다: {model_file}")

    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    if not hasattr(model_module, "MyModel"):
        raise AttributeError(f"{model_file} 안에 MyModel 클래스가 없습니다.")
    return model_module.MyModel


# ==========================================================
# [CT 전처리]
# ==========================================================
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if USE_CLAHE:
        img = _clahe.apply(img)
    return img


def load_25d_or_repeat(slice_path, prev_path, next_path):
    cur = load_gray(slice_path)
    if cur is None:
        return None
    if not USE_25D:
        return np.stack([cur, cur, cur], axis=-1)

    prev = load_gray(prev_path) if (prev_path and os.path.exists(prev_path)) else cur
    nxt  = load_gray(next_path) if (next_path and os.path.exists(next_path)) else cur

    if prev.shape != cur.shape:
        prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]))
    if nxt.shape != cur.shape:
        nxt = cv2.resize(nxt, (cur.shape[1], cur.shape[0]))

    return np.stack([prev, cur, nxt], axis=-1)


# ==========================================================
# [Letterbox resize — 비율 유지 + 검정 패딩]
#
# ★ train_crop_2head.py / inference_crop_2head.py 양쪽에서 완전히 동일한 코드 사용.
#   스케일 방식(max 기준), 패딩 색(검정=0), 중앙 정렬 방식이 다르면
#   학습/추론 분포가 어긋나 점수가 무너짐. 절대 한쪽만 수정하지 말 것.
# ==========================================================
def letterbox_square(img_3ch, size):
    h, w   = img_3ch.shape[:2]
    scale  = size / max(h, w)
    new_h  = round(h * scale)
    new_w  = round(w * scale)

    resized = cv2.resize(img_3ch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_y  = (size - new_h) // 2
    pad_x  = (size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


# ==========================================================
# [데이터셋 mean/std 계산 — crop 적용 후 기준]
# ==========================================================
def compute_dataset_stats_crop(samples, max_n=2000):
    rng  = random.Random(SEED)
    pick = samples if len(samples) <= max_n else rng.sample(samples, max_n)
    means, stds = [], []

    for s in tqdm(pick, desc="dataset stats (crop)"):
        x = load_25d_or_repeat(s["path"], s.get("prev"), s.get("next"))
        if x is None:
            continue
        h, w = x.shape[:2]
        # resize 전에 crop 적용 (학습 시 __getitem__과 동일 순서)
        x = x[:, :w // 2] if s["side"] == "left" else x[:, w // 2:]
        x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        means.append(x.mean(axis=(0, 1)))
        stds.append(x.std(axis=(0, 1)))

    if len(means) == 0:
        raise RuntimeError("mean/std 계산에 사용할 이미지가 없습니다.")

    mean = np.mean(means, axis=0).tolist()
    std  = np.mean(stds, axis=0).tolist()
    std  = [max(v, 1e-3) for v in std]
    return mean, std


# ==========================================================
# [CSV 중복 행 제거]
# ==========================================================
def dedupe_rows(df):
    original_n = len(df)
    df_work    = df.copy()
    first_col  = df_work.columns[0]
    second_col = df_work.columns[1]

    keep_indices  = []
    removed_count = 0

    for key, group in df_work.groupby([first_col, second_col], sort=False, dropna=False):
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
        seen_rows = {}
        for ridx, row in group.iterrows():
            sig = tuple("NaN" if pd.isna(v) else v for v in row.tolist())
            if sig not in seen_rows:
                seen_rows[sig] = ridx
                keep_indices.append(ridx)
            else:
                removed_count += 1

    cleaned = df_work.loc[keep_indices].reset_index(drop=True)
    print(f"[중복 제거] {original_n}행 → {len(cleaned)}행 (제거 {removed_count}행)")
    return cleaned


# ==========================================================
# [원본 4-태스크 인덱스 판별 — 내부 라벨 수집용]
# ==========================================================
def _orig_task_index(rl_value, image_number_value):
    """CSV R/L + Image number → [rt_temporal=0, lt_temporal=1, rt_otitis=2, lt_otitis=3]"""
    rl  = str(rl_value).strip().lower()
    img = str(image_number_value).strip().lower()
    is_rt       = "rt" in rl
    is_temporal = "temporal" in img
    if is_rt and is_temporal:         return 0
    if (not is_rt) and is_temporal:   return 1
    if is_rt and (not is_temporal):   return 2
    return 3


# ==========================================================
# [샘플 리스트 구축 — 좌/우 crop 분리, 2-slot 라벨]
#
#   1단계: (환자, 슬라이스)마다 4개 라벨 수집
#          [rt_temporal, lt_temporal, rt_otitis, lt_otitis]
#   2단계: CROP_SIDE_TO_RL에 따라 좌/우 2개 샘플로 분리
#          각 crop 샘플 라벨: [temporal, otitis] (shape=(2,))
#            - LEFT  crop (standard=환자Rt): [rt_temporal, rt_otitis]
#            - RIGHT crop (standard=환자Lt): [lt_temporal, lt_otitis]
#          ★ IGNORE 없음. otitis head를 left/right가 공유하므로
#            양쪽 crop 모두 temporal·otitis 두 슬롯에 유효 라벨이 들어간다.
# ==========================================================
def build_sample_list_crop(csv_path, base_path):
    df = pd.read_csv(csv_path)
    print(f"데이터 로드: {csv_path} ({len(df)}행)")
    df = dedupe_rows(df)

    rl_col     = "R/L"
    img_col    = "Image number"
    slice_cols = [str(i) for i in range(1, 133)]

    # ── 1단계: 4-라벨 수집 ──
    raw_dict = {}  # (pid, slice_n) → dict

    print("4-label 수집 중 (crop 분리 전)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        p_id     = str(int(row["No"]))
        task_idx = _orig_task_index(row[rl_col], row[img_col])

        p_dir = os.path.join(base_path, p_id)
        if not os.path.exists(p_dir):
            p_dir = os.path.join(base_path, "train", p_id)
        img_dir = os.path.join(p_dir, "PNG_soft")
        if not os.path.exists(img_dir):
            continue

        for col in slice_cols:
            if col not in row.index or pd.isna(row[col]):
                continue
            label = int(row[col])
            n     = int(col)
            cur   = os.path.join(img_dir, f"{n:04d}.png")
            if not os.path.exists(cur):
                continue

            key = (p_id, n)
            if key not in raw_dict:
                prev = os.path.join(img_dir, f"{n-1:04d}.png") if n > 1 else None
                nxt  = os.path.join(img_dir, f"{n+1:04d}.png") if n < 132 else None
                raw_dict[key] = {
                    "path":     cur,
                    "prev":     prev,
                    "next":     nxt,
                    "labels_4": [IGNORE_LABEL] * 4,  # [rt_temp, lt_temp, rt_ot, lt_ot]
                    "pid":      p_id,
                }
            raw_dict[key]["labels_4"][task_idx] = label

    # ── 2단계: 좌/우 crop 샘플 분리 (2-slot 라벨) ──
    # 슬롯: [0]=temporal, [1]=otitis (좌우 crop이 otitis head를 공유)
    # labels_4 인덱스: rt_temporal=0, lt_temporal=1, rt_otitis=2, lt_otitis=3
    if CROP_SIDE_TO_RL == "standard":
        left_t,  left_o  = 0, 2   # LEFT  crop → 환자 Rt: temporal=rt_temp, otitis=rt_otitis
        right_t, right_o = 1, 3   # RIGHT crop → 환자 Lt: temporal=lt_temp, otitis=lt_otitis
    else:
        left_t,  left_o  = 1, 3   # LEFT  crop → 환자 Lt
        right_t, right_o = 0, 2   # RIGHT crop → 환자 Rt

    samples = []
    for info in raw_dict.values():
        lbl4 = info["labels_4"]
        base = {k: info[k] for k in ("path", "prev", "next", "pid")}

        left_labels  = [lbl4[left_t],  lbl4[left_o]]   # [temporal, otitis] — IGNORE 없음
        right_labels = [lbl4[right_t], lbl4[right_o]]  # [temporal, otitis] — IGNORE 없음

        # 유효 라벨이 하나라도 있는 경우만 샘플 생성
        if any(l >= 0 for l in left_labels):
            samples.append({**base, "side": "left",  "labels": left_labels})
        if any(l >= 0 for l in right_labels):
            samples.append({**base, "side": "right", "labels": right_labels})

    print(f"총 crop 샘플 수: {len(samples)}  "
          f"(슬라이스 {len(raw_dict)}개 × 좌우 분리, "
          f"CROP_SIDE_TO_RL='{CROP_SIDE_TO_RL}')")

    if len(samples) == 0:
        raise RuntimeError("생성된 학습 샘플이 없습니다.")

    for t, name in enumerate(TASK_NAMES):
        labels_t = [s["labels"][t] for s in samples if s["labels"][t] >= 0]
        if labels_t:
            cnt = np.bincount(labels_t, minlength=2)
            print(f"  {name}: Normal={cnt[0]}, Abnormal={cnt[1]}")
        else:
            print(f"  {name}: 라벨 없음")

    return samples


# ==========================================================
# [Dataset — crop 버전, side 인덱스 포함 3-tuple 반환]
# ==========================================================
class TBCTDatasetCrop(Dataset):
    def __init__(self, samples, mean, std, training=False):
        self.samples  = samples
        self.training = training
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _augment(self, img_3ch):
        """crop 이후 IMAGE_SIZE 정사각형 이미지에 적용. 좌우 flip 없음."""
        h, w = img_3ch.shape[:2]

        if random.random() < 0.8:
            angle = random.uniform(-15, 15)
            tx    = random.uniform(-0.07, 0.07) * w
            ty    = random.uniform(-0.07, 0.07) * h
            scale = random.uniform(0.90, 1.10)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img_3ch = cv2.warpAffine(
                img_3ch, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        if random.random() < 0.6:
            alpha = random.uniform(0.80, 1.20)
            beta  = random.uniform(-15, 15)
            img_3ch = np.clip(
                img_3ch.astype(np.float32) * alpha + beta, 0, 255
            ).astype(np.uint8)

        if random.random() < 0.40:
            eh = random.randint(int(0.05 * h), int(0.20 * h))
            ew = random.randint(int(0.05 * w), int(0.20 * w))
            y0 = random.randint(0, h - eh)
            x0 = random.randint(0, w - ew)
            img_3ch[y0:y0+eh, x0:x0+ew, :] = 0

        return img_3ch

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = load_25d_or_repeat(s["path"], s.get("prev"), s.get("next"))
        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE * 2, 3), dtype=np.uint8)

        # ① 좌우 crop (resize 전)
        h, w = img.shape[:2]
        img = img[:, :w // 2] if s["side"] == "left" else img[:, w // 2:]

        # ② 강제 resize → IMAGE_SIZE 정사각형
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        # ③ 증강 (training=True일 때만)
        if self.training:
            img = self._augment(img)

        # ④ normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))

        labels   = torch.tensor(s["labels"], dtype=torch.long)
        side_idx = 0 if s["side"] == "left" else 1

        return torch.from_numpy(img.copy()), labels, side_idx


# ==========================================================
# [FocalLoss]
# ==========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        if self.ls > 0:
            n_cls = logits.size(1)
            with torch.no_grad():
                true = torch.zeros_like(logp).fill_(self.ls / (n_cls - 1))
                true.scatter_(1, target.unsqueeze(1), 1.0 - self.ls)
            ce = -(true * logp).sum(dim=1)
        else:
            ce = F.nll_loss(logp, target, reduction="none")

        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            at = self.alpha.to(logits.device)[target]
            focal = at * focal

        return focal.mean()


# ==========================================================
# [Multi-Head Loss — baseline (합산 가중)]
#
#   outputs:  (B, 2, 2)
#   labels:   (B, 2)    (-1이면 무시)
#   criteria: list of 2 loss functions
# ==========================================================
def compute_multi_loss(outputs, labels, criteria):
    total_loss  = 0.0
    valid_tasks = 0

    for t in range(NUM_TASKS):
        mask = labels[:, t] >= 0
        if mask.sum() == 0:
            continue
        total_loss += criteria[t](outputs[mask, t, :], labels[mask, t])
        valid_tasks += 1

    if valid_tasks == 0:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    return total_loss / valid_tasks


# ==========================================================
# [Multi-Head Loss — side-resolved otitis 가중]
#
#   outputs:  (B, 2, 2)
#   labels:   (B, 2)         (-1이면 무시)
#   sides:    (B,) LongTensor  0=left, 1=right
#   criteria: dict {"temporal", "otitis_left", "otitis_right"}
#
#   temporal head: 양쪽 합산 (baseline과 동일)
#   otitis head  : left/right 분리 가중 → 두 그룹 loss 평균 후 head 1개분으로 산입
# ==========================================================
def compute_multi_loss_sideweighted(outputs, labels, sides, criteria):
    total_loss  = 0.0
    valid_heads = 0

    # head 0: temporal (양쪽 합산) — baseline과 동일
    m = labels[:, 0] >= 0
    if m.sum() > 0:
        total_loss += criteria["temporal"](outputs[m, 0, :], labels[m, 0])
        valid_heads += 1

    # head 1: otitis — 좌/우 분리 가중 후 평균 → 'head 1개분'으로 환산
    ot_loss, ot_groups = 0.0, 0
    for side_idx, key in [(0, "otitis_left"), (1, "otitis_right")]:
        mm = (labels[:, 1] >= 0) & (sides == side_idx)
        if mm.sum() > 0:
            ot_loss += criteria[key](outputs[mm, 1, :], labels[mm, 1])
            ot_groups += 1
    if ot_groups > 0:
        total_loss += ot_loss / ot_groups
        valid_heads += 1

    if valid_heads == 0:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    return total_loss / valid_heads


# ==========================================================
# [Side-resolved criteria 빌더]
#
#   temporal: 양쪽 합산 클래스 분포로 FocalLoss 생성
#   otitis_left / otitis_right: side별 분포로 각각 FocalLoss 생성
# ==========================================================
def build_criteria_sideweighted(samples):
    def focal_from_counts(cnt):
        cw = torch.tensor(1.0 / np.maximum(cnt, 1), dtype=torch.float32)
        cw = cw / cw.sum() * len(cw)
        return FocalLoss(alpha=cw.to(DEVICE), gamma=2.0, label_smoothing=0.05)

    temp = [s["labels"][0] for s in samples if s["labels"][0] >= 0]
    ol   = [s["labels"][1] for s in samples if s["side"] == "left"  and s["labels"][1] >= 0]
    orr  = [s["labels"][1] for s in samples if s["side"] == "right" and s["labels"][1] >= 0]

    c_t, c_ol, c_or = (np.bincount(temp, minlength=2),
                        np.bincount(ol,   minlength=2),
                        np.bincount(orr,  minlength=2))

    print(f"  temporal:         Normal={c_t[0]},  Abnormal={c_t[1]}")
    print(f"  otitis(left=rt):  Normal={c_ol[0]}, Abnormal={c_ol[1]}")
    print(f"  otitis(right=lt): Normal={c_or[0]}, Abnormal={c_or[1]}")

    return {
        "temporal":     focal_from_counts(c_t),
        "otitis_left":  focal_from_counts(c_ol),
        "otitis_right": focal_from_counts(c_or),
    }


# ==========================================================
# [Validation: head별 softmax 확률 수집]
# ==========================================================
@torch.no_grad()
def collect_preds_prob(model, loader):
    model.eval()
    task_probs  = [[] for _ in range(NUM_TASKS)]
    task_labels = [[] for _ in range(NUM_TASKS)]

    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        outputs = model(x)                              # (B, 2, 2)
        probs   = torch.softmax(outputs, dim=2)[:, :, 1]  # (B, 2)

        for t in range(NUM_TASKS):
            mask = y[:, t] >= 0
            if mask.sum() > 0:
                task_probs[t].extend(probs[mask, t].cpu().numpy())
                task_labels[t].extend(y[mask, t].cpu().numpy())

    return (
        [np.array(p) for p in task_probs],
        [np.array(l) for l in task_labels],
    )


# ==========================================================
# [KFold split 준비 — 환자 단위 그룹 보장]
# ==========================================================
def build_fold_splits(all_samples):
    """같은 환자의 left/right crop이 항상 같은 fold에 들어가도록 GroupKFold."""
    pid_labels: dict = {}
    for s in all_samples:
        pid = s["pid"]
        if pid not in pid_labels:
            pid_labels[pid] = []
        pid_labels[pid].extend(l for l in s["labels"] if l >= 0)

    pids_sorted = sorted(pid_labels.keys())
    pid_ratio   = {p: float(np.mean(v)) if v else 0.0 for p, v in pid_labels.items()}
    ratio_arr   = np.array([pid_ratio[p] for p in pids_sorted])
    try:
        pid_bin_arr = pd.qcut(ratio_arr, q=3, labels=False, duplicates="drop")
    except ValueError:
        pid_bin_arr = np.zeros(len(pids_sorted), dtype=int)
    pid_bin = {p: int(b) for p, b in zip(pids_sorted, pid_bin_arr)}

    groups     = np.array([s["pid"] for s in all_samples])
    sample_bin = np.array([pid_bin.get(s["pid"], 0) for s in all_samples])

    kfold = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    return list(kfold.split(all_samples, sample_bin, groups))


# ==========================================================
# [디버깅: crop 이미지 저장 후 종료]
# ==========================================================
def dump_crops(data_root, train_csv, n_patients=4, out_dir="./crop_2head_debug"):
    os.makedirs(out_dir, exist_ok=True)
    all_samples = build_sample_list_crop(train_csv, data_root)

    # 첫 n_patients명 선정 (등장 순서 기준)
    seen: list = []
    for s in all_samples:
        if s["pid"] not in seen:
            seen.append(s["pid"])
        if len(seen) >= n_patients:
            break
    target_pids = set(seen)

    saved = 0
    for s in all_samples:
        if s["pid"] not in target_pids:
            continue
        img = load_25d_or_repeat(s["path"], s.get("prev"), s.get("next"))
        if img is None:
            continue
        h, w = img.shape[:2]
        crop  = img[:, :w // 2] if s["side"] == "left" else img[:, w // 2:]
        # 중간 채널(현재 슬라이스)을 그레이스케일로 저장
        out_img = crop[:, :, 1]
        slice_n = int(os.path.splitext(os.path.basename(s["path"]))[0])
        fname   = f"p{s['pid']}_s{slice_n:03d}_{s['side']}.png"
        cv2.imwrite(os.path.join(out_dir, fname), out_img)
        saved += 1

    print(f"\nCrop 디버그 이미지 {saved}장 저장 완료: {out_dir}/")
    print(f"  확인 환자: {sorted(target_pids)}")
    print(f"  CROP_SIDE_TO_RL='{CROP_SIDE_TO_RL}' 기준 — "
          f"LEFT=환자{'Rt' if CROP_SIDE_TO_RL=='standard' else 'Lt'}, "
          f"RIGHT=환자{'Lt' if CROP_SIDE_TO_RL=='standard' else 'Rt'}")


# ==========================================================
# [학습 메인 — K-Fold]
# ==========================================================
def train():
    DATA_ROOT = "./data"
    if not os.path.exists(DATA_ROOT):
        DATA_ROOT = "../data"
    TRAIN_CSV = "train_set.csv"
    if not os.path.exists(TRAIN_CSV):
        TRAIN_CSV = "../train_set.csv"

    model_name         = os.path.splitext(os.path.basename(MODEL_FILE))[0]
    preproc_stats_path = f"preproc_{model_name}_stats.json"

    print(f"사용 디바이스: {DEVICE}")
    print(f"사용 모델 파일: {MODEL_FILE}")
    print(f"CROP_SIDE_TO_RL: '{CROP_SIDE_TO_RL}'")
    print(f"저장될 파일: best_{model_name}_fold{{k}}.pth (k=0..{NUM_FOLDS-1}), "
          f"{preproc_stats_path}")
    print(f"옵션: CLAHE={USE_CLAHE}, 2.5D={USE_25D}, FocalLoss={USE_FOCAL_LOSS}, "
          f"NUM_FOLDS={NUM_FOLDS}, NUM_TASKS={NUM_TASKS}(crop, 2-head shared-otitis, side-resolved loss + EMA)")

    MyModel = load_model_class(MODEL_FILE)
    print(f"모델 로드 완료: {MODEL_FILE} → MyModel")

    all_samples = build_sample_list_crop(TRAIN_CSV, DATA_ROOT)

    # ── mean/std: crop 적용 후 기준으로 1회 계산 ──
    print("데이터셋 mean/std 계산 중 (crop 적용 기준)...")
    mean, std = compute_dataset_stats_crop(all_samples)
    print(f"  mean={mean}")
    print(f"  std ={std}")

    with open(preproc_stats_path, "w") as f:
        json.dump({
            "mean": mean, "std": std,
            "use_clahe": USE_CLAHE, "use_25d": USE_25D,
            "image_size": IMAGE_SIZE, "model_file": MODEL_FILE,
            "num_tasks": NUM_TASKS, "crop_side_to_rl": CROP_SIDE_TO_RL,
        }, f, indent=2)

    fold_splits = build_fold_splits(all_samples)

    # ── K-Fold 학습 루프 ──
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"[Fold {fold_idx+1}/{NUM_FOLDS}]")
        print(f"{'='*60}")

        fold_model_path = f"best_{model_name}_fold{fold_idx}.pth"

        train_samples = [all_samples[i] for i in train_idx]
        val_samples   = [all_samples[i] for i in val_idx]
        print(f"Train 샘플: {len(train_samples)} | Val 샘플: {len(val_samples)}")

        # 전체 분포 출력 (참고용)
        fold_train_labels = [l for s in train_samples for l in s["labels"] if l >= 0]
        cls_count_all = np.bincount(fold_train_labels, minlength=2)
        pos_ratio = cls_count_all[1] / max(cls_count_all.sum(), 1)
        print(f"Train 클래스 분포(전체): Normal={cls_count_all[0]}, Abnormal={cls_count_all[1]} "
              f"(병변 비율 {pos_ratio*100:.1f}%)")

        train_ds = TBCTDatasetCrop(train_samples, mean, std, training=True)
        val_ds   = TBCTDatasetCrop(val_samples,   mean, std, training=False)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=False, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=False,
        )

        model = MyModel().to(DEVICE)
        ema   = ModelEMA(model, decay=0.999)

        # ── side-resolved criteria ──
        print("Loss: side-resolved FocalLoss (temporal=합산, otitis=left/right 분리)")
        criteria = build_criteria_sideweighted(train_samples)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        use_amp = torch.cuda.is_available()
        scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_ema_f1 = -1.0
        no_improve  = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            all_preds_flat, all_labels_flat = [], []

            print(f"\n[Fold {fold_idx+1} Epoch {epoch}/{NUM_EPOCHS}]  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            pbar = tqdm(train_loader, unit="batch")

            for inputs, labels, sides in pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                sides  = sides.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)                        # (B, 2, 2)
                    loss    = compute_multi_loss_sideweighted(outputs, labels, sides, criteria)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                running_loss += loss.item() * inputs.size(0)
                n_seen += inputs.size(0)

                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=2)
                    for t in range(NUM_TASKS):
                        mask = labels[:, t] >= 0
                        if mask.sum() > 0:
                            all_preds_flat.extend(preds[mask, t].cpu().numpy())
                            all_labels_flat.extend(labels[mask, t].cpu().numpy())

                pbar.set_postfix(loss=loss.item())

            scheduler.step()

            train_loss = running_loss / max(n_seen, 1)
            train_acc  = accuracy_score(all_labels_flat, all_preds_flat)
            train_f1   = f1_score(all_labels_flat, all_preds_flat,
                                  average="macro", zero_division=0)

            # ── Validation: raw 모델 argmax(0.5) 기준 F1 ──
            val_task_probs, val_task_labels = collect_preds_prob(model, val_loader)

            vp_flat, vl_flat = [], []
            for t in range(NUM_TASKS):
                if len(val_task_labels[t]) > 0:
                    vp_flat.extend((val_task_probs[t] >= 0.5).astype(int))
                    vl_flat.extend(val_task_labels[t])

            if len(vl_flat) > 0:
                f1_e    = f1_score(vl_flat, vp_flat, average="macro", zero_division=0)
                val_acc = accuracy_score(vl_flat, vp_flat)
            else:
                f1_e    = -1.0
                val_acc = 0.0

            # ── Validation: EMA 모델 argmax(0.5) 기준 F1 ──
            ema_task_probs, ema_task_labels = collect_preds_prob(ema.ema, val_loader)
            evp_flat, evl_flat = [], []
            for t in range(NUM_TASKS):
                if len(ema_task_labels[t]) > 0:
                    evp_flat.extend((ema_task_probs[t] >= 0.5).astype(int))
                    evl_flat.extend(ema_task_labels[t])
            f1_ema = f1_score(evl_flat, evp_flat, average="macro", zero_division=0) if evl_flat else -1.0

            print(f"Train Loss {train_loss:.4f} | F1(train) {train_f1:.4f}  [Acc {train_acc:.4f}]")
            print(f"Val   F1(macro) raw={f1_e:.4f} | EMA={f1_ema:.4f} @ argmax")

            if f1_ema > best_ema_f1:
                best_ema_f1 = f1_ema
                no_improve  = 0
                torch.save(ema.ema.state_dict(), fold_model_path)
                print(f"  >> Best EMA F1 갱신: {best_ema_f1:.4f} → {fold_model_path} 저장")
            else:
                no_improve += 1
                print(f"  (no improvement {no_improve}/{PATIENCE})")
                if no_improve >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"\n[Fold {fold_idx+1}] 완료. Best EMA Val F1 = {best_ema_f1:.4f}")

    print(f"\n학습 완료. 저장 파일: best_{model_name}_fold{{k}}.pth, {preproc_stats_path}")


# ==========================================================
# [Full-fit — 전체 데이터, EarlyStopping 없음]
# ==========================================================
def train_fullfit(fullfit_epochs):
    DATA_ROOT = "./data"
    if not os.path.exists(DATA_ROOT):
        DATA_ROOT = "../data"
    TRAIN_CSV = "train_set.csv"
    if not os.path.exists(TRAIN_CSV):
        TRAIN_CSV = "../train_set.csv"

    model_name         = os.path.splitext(os.path.basename(MODEL_FILE))[0]
    fullfit_path       = f"best_{model_name}_fullfit.pth"
    preproc_stats_path = f"preproc_{model_name}_stats.json"

    print(f"사용 디바이스: {DEVICE}")
    print(f"사용 모델 파일: {MODEL_FILE}")
    print(f"CROP_SIDE_TO_RL: '{CROP_SIDE_TO_RL}'")
    print(f"저장될 파일: {fullfit_path}, {preproc_stats_path}")
    print(f"학습 에폭: {fullfit_epochs}  (--fullfit-epochs로 변경 가능)")
    print(f"옵션: CLAHE={USE_CLAHE}, 2.5D={USE_25D}, FocalLoss={USE_FOCAL_LOSS}, "
          f"drop_last=True, NUM_TASKS={NUM_TASKS}(crop, 2-head shared-otitis, side-resolved loss + EMA)")

    MyModel = load_model_class(MODEL_FILE)

    all_samples = build_sample_list_crop(TRAIN_CSV, DATA_ROOT)
    print(f"전체 crop 샘플: {len(all_samples)}")

    print("데이터셋 mean/std 계산 중 (crop 적용 기준)...")
    mean, std = compute_dataset_stats_crop(all_samples)
    print(f"  mean={mean}")
    print(f"  std ={std}")

    with open(preproc_stats_path, "w") as f:
        json.dump({
            "mean": mean, "std": std,
            "use_clahe": USE_CLAHE, "use_25d": USE_25D,
            "image_size": IMAGE_SIZE, "model_file": MODEL_FILE,
            "num_tasks": NUM_TASKS, "crop_side_to_rl": CROP_SIDE_TO_RL,
        }, f, indent=2)

    # 전체 분포 출력 (참고용)
    all_labels_flat_all = [l for s in all_samples for l in s["labels"] if l >= 0]
    cls_count_all = np.bincount(all_labels_flat_all, minlength=2)
    pos_ratio = cls_count_all[1] / max(cls_count_all.sum(), 1)
    print(f"클래스 분포(전체): Normal={cls_count_all[0]}, Abnormal={cls_count_all[1]} "
          f"(병변 비율 {pos_ratio*100:.1f}%)")

    train_ds = TBCTDatasetCrop(all_samples, mean, std, training=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=True,
    )

    model = MyModel().to(DEVICE)
    ema   = ModelEMA(model, decay=0.999)

    # ── side-resolved criteria ──
    print("Loss: side-resolved FocalLoss (temporal=합산, otitis=left/right 분리)")
    criteria = build_criteria_sideweighted(all_samples)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=fullfit_epochs, eta_min=1e-6,
    )

    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, fullfit_epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        all_preds_flat, all_labels_flat = [], []

        print(f"\n[Fullfit Epoch {epoch}/{fullfit_epochs}]  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        pbar = tqdm(train_loader, unit="batch")

        for inputs, labels, sides in pbar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            sides  = sides.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss    = compute_multi_loss_sideweighted(outputs, labels, sides, criteria)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += loss.item() * inputs.size(0)
            n_seen += inputs.size(0)

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=2)
                for t in range(NUM_TASKS):
                    mask = labels[:, t] >= 0
                    if mask.sum() > 0:
                        all_preds_flat.extend(preds[mask, t].cpu().numpy())
                        all_labels_flat.extend(labels[mask, t].cpu().numpy())

            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        train_loss = running_loss / max(n_seen, 1)
        train_f1   = f1_score(all_labels_flat, all_preds_flat,
                              average="macro", zero_division=0)
        train_acc  = accuracy_score(all_labels_flat, all_preds_flat)
        print(f"Train Loss {train_loss:.4f} | F1(train) {train_f1:.4f}  [Acc {train_acc:.4f}]")

    torch.save(ema.ema.state_dict(), fullfit_path)
    print(f"\nFullfit 완료 (EMA 저장). 저장: {fullfit_path}  ({fullfit_epochs}에폭)")


# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TBCT Crop 2-head side-resolved otitis loss + EMA 학습 스크립트 (model_convNeXt-Tiny_crop_2head_sideloss_ema.py 전용)",
        usage="python3 train_crop_2head_sideloss_ema.py [--fullfit] [--fullfit-epochs N] [--dump-crops]",
    )
    parser.add_argument("--fullfit", action="store_true",
                        help="전체 데이터로 단일 모델 학습 (val/EarlyStopping 없음)")
    parser.add_argument("--fullfit-epochs", type=int, default=NUM_EPOCHS,
                        help=f"fullfit 학습 에폭 수 (기본={NUM_EPOCHS})")
    parser.add_argument("--dump-crops", action="store_true",
                        help="crop 디버그 이미지를 ./crop_2head_debug/에 저장 후 종료 (학습 안 함)")

    args = parser.parse_args()

    DATA_ROOT = "./data" if os.path.exists("./data") else "../data"
    TRAIN_CSV = "train_set.csv" if os.path.exists("train_set.csv") else "../train_set.csv"

    if args.dump_crops:
        dump_crops(DATA_ROOT, TRAIN_CSV)
    elif args.fullfit:
        train_fullfit(args.fullfit_epochs)
    else:
        train()
