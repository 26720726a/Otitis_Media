import os
import json
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


# ==========================================================
# [모델 파일 고정]
# ★ 경고: MODEL_FILE이 train_crop_2head.py와 동일하므로
#   체크포인트(best_model_convNeXt-Tiny_crop_2head_fold*.pth)를
#   학습 전에 반드시 백업해 두세요. 덮어쓰기됩니다.
# ==========================================================
MODEL_FILE = "model_convNeXt-Tiny_crop_2head.py"


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
# [WeightedRandomSampler 설정]
#
# SAMPLER_MODE:
#   "target_ratio"  — rt 양성(G1)만 목표 비율로 끌어올리고
#                     나머지 그룹(G2·G3·G4)은 원본 상대 비율 유지.
#                     lt 분포를 거의 건드리지 않음. (기본)
#   "inverse_freq"  — 4그룹 전부 1/count 역빈도 가중치. (비교 실험용)
#
# RT_POS_TARGET_RATIO: G1(rt 양성 crop)이 배치에서 차지할 목표 비율.
#   기본 0.30. 원본 ~15% → 0.30으로 약 2배 오버샘플.
# ==========================================================
SAMPLER_MODE        = "target_ratio"   # "target_ratio" | "inverse_freq"
RT_POS_TARGET_RATIO = 0.30             # G1(rt 양성) 목표 등장 비율


# ==========================================================
# [좌우 crop 방향 설정]
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
# ★ train_crop_2head_sampler.py / inference_crop_2head.py 양쪽에서 완전히 동일한 코드 사용.
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
# [Dataset — crop 버전]
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

        labels = torch.tensor(s["labels"], dtype=torch.long)  # (2,)

        return torch.from_numpy(img.copy()), labels


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
# [Multi-Head Loss — 2-head, head별 독립 criterion]
#
#   outputs:  (B, 2, 2)
#   labels:   (B, 2)    (-1이면 무시)
#   criteria: list of 2 loss functions, criteria[t] for head t
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
# [Validation: head별 softmax 확률 수집]
# ==========================================================
@torch.no_grad()
def collect_preds_prob(model, loader):
    model.eval()
    task_probs  = [[] for _ in range(NUM_TASKS)]
    task_labels = [[] for _ in range(NUM_TASKS)]

    for x, y in loader:
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
# [WeightedRandomSampler 생성]
#
# 4그룹 분류 (CROP_SIDE_TO_RL에 따라 rt_side 결정):
#   G1 = rt_side & otitis==1  (rt 양성) ← 최우선 오버샘플
#   G2 = rt_side & otitis==0  (rt 음성)
#   G3 = lt_side & otitis==1  (lt 양성)
#   G4 = lt_side & otitis==0  (lt 음성)
#   otitis==IGNORE_LABEL → weight=1.0 (방어적, 2-head에서는 발생 안 함)
#
# "target_ratio" 모드:
#   - G1 목표비율 = RT_POS_TARGET_RATIO
#   - G2·G3·G4는 원본 상대 비율 유지 (lt 과대표집 방지)
#   - weight_G1 = target * total / count_G1
#   - weight_G2..G4 = (1-target) * total / (count_G2+count_G3+count_G4)
#
# "inverse_freq" 모드:
#   - 4그룹 전부 weight = total / count_G
# ==========================================================
def make_sampler(samples, mode=SAMPLER_MODE):
    n = len(samples)

    # rt_side: CROP_SIDE_TO_RL='standard' → LEFT=환자Rt
    rt_side = "left"  if CROP_SIDE_TO_RL == "standard" else "right"
    lt_side = "right" if CROP_SIDE_TO_RL == "standard" else "left"

    G1, G2, G3, G4, G_IGNORE = 0, 1, 2, 3, -1
    group_names = [
        f"G1({rt_side.upper()} & otitis=1, rt양성)",
        f"G2({rt_side.upper()} & otitis=0, rt음성)",
        f"G3({lt_side.upper()} & otitis=1, lt양성)",
        f"G4({lt_side.upper()} & otitis=0, lt음성)",
    ]

    # 샘플별 그룹 분류
    groups = np.full(n, G_IGNORE, dtype=np.int32)
    for i, s in enumerate(samples):
        ot   = s["labels"][1]   # otitis slot (index 1)
        side = s["side"]
        if ot == IGNORE_LABEL:
            groups[i] = G_IGNORE
        elif side == rt_side and ot == 1:
            groups[i] = G1
        elif side == rt_side and ot == 0:
            groups[i] = G2
        elif side == lt_side and ot == 1:
            groups[i] = G3
        else:
            groups[i] = G4

    counts = {g: int((groups == g).sum()) for g in (G1, G2, G3, G4)}
    n_ignore    = int((groups == G_IGNORE).sum())
    total_known = n - n_ignore

    # 원본 분포 출력
    print(f"  [Sampler] mode={mode!r} | CROP_SIDE_TO_RL={CROP_SIDE_TO_RL!r} | "
          f"total={n}, ignore={n_ignore}")
    for g, name in enumerate(group_names):
        orig_frac = counts[g] / max(total_known, 1)
        print(f"    {name}: count={counts[g]:4d}  원본비율={orig_frac*100:5.1f}%")

    # 그룹별 가중치 계산
    if mode == "target_ratio":
        total_non_g1 = counts[G2] + counts[G3] + counts[G4]
        w_g1     = RT_POS_TARGET_RATIO * total_known / max(counts[G1], 1)
        w_non_g1 = (1.0 - RT_POS_TARGET_RATIO) * total_known / max(total_non_g1, 1)
        group_w  = {G1: w_g1, G2: w_non_g1, G3: w_non_g1, G4: w_non_g1}
    else:  # "inverse_freq"
        group_w = {g: total_known / max(counts[g], 1) for g in (G1, G2, G3, G4)}

    # 샘플별 가중치 배열
    weights = np.ones(n, dtype=np.float64)
    for i, g in enumerate(groups):
        if g != G_IGNORE:
            weights[i] = group_w[g]

    # 적용 후 기대 등장비율 출력 (가중치 합 기준)
    total_w = weights.sum()
    print(f"  [Sampler] 적용 후 기대 등장비율 (RT_POS_TARGET_RATIO={RT_POS_TARGET_RATIO}):")
    for g, name in enumerate(group_names):
        mask     = groups == g
        expected = weights[mask].sum() / max(total_w, 1)
        print(f"    {name}: 기대비율={expected*100:5.1f}%  "
              f"(가중치={group_w[g]:.3f})")

    g1_expected = weights[groups == G1].sum() / max(total_w, 1)
    if abs(g1_expected - RT_POS_TARGET_RATIO) < 0.02:
        print(f"  [OK] G1(rt양성) 기대비율={g1_expected*100:.1f}% → "
              f"목표 {RT_POS_TARGET_RATIO*100:.0f}% 달성 ✓")
    else:
        print(f"  [주의] G1(rt양성) 기대비율={g1_expected*100:.1f}% "
              f"(목표 {RT_POS_TARGET_RATIO*100:.0f}%와 2%p 이상 차이)")

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=n,
        replacement=True,
    )


# ==========================================================
# [디버깅: crop 이미지 저장 후 종료]
# ==========================================================
def dump_crops(data_root, train_csv, n_patients=4, out_dir="./crop_2head_sampler_debug"):
    os.makedirs(out_dir, exist_ok=True)
    all_samples = build_sample_list_crop(train_csv, data_root)

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

    # ── 체크포인트 덮어쓰기 경고 ──
    existing = [f for f in (f"best_{model_name}_fold{k}.pth" for k in range(NUM_FOLDS))
                if os.path.exists(f)]
    if existing:
        print("\n" + "!" * 60)
        print("[경고] 아래 파일이 이미 존재합니다. 학습 시 덮어쓰기됩니다:")
        for fp in existing:
            print(f"  {fp}")
        print("  train_crop_2head.py의 체크포인트와 공유하므로")
        print("  필요하다면 지금 백업하세요. (Ctrl+C로 중단 가능)")
        print("!" * 60 + "\n")

    print(f"사용 디바이스: {DEVICE}")
    print(f"사용 모델 파일: {MODEL_FILE}")
    print(f"CROP_SIDE_TO_RL: '{CROP_SIDE_TO_RL}'")
    print(f"Sampler: mode={SAMPLER_MODE!r}, RT_POS_TARGET_RATIO={RT_POS_TARGET_RATIO}")
    print(f"저장될 파일: best_{model_name}_fold{{k}}.pth (k=0..{NUM_FOLDS-1}), "
          f"{preproc_stats_path}")
    print(f"옵션: CLAHE={USE_CLAHE}, 2.5D={USE_25D}, FocalLoss={USE_FOCAL_LOSS}, "
          f"NUM_FOLDS={NUM_FOLDS}, NUM_TASKS={NUM_TASKS}(crop, 2-head + WeightedSampler)")

    MyModel = load_model_class(MODEL_FILE)
    print(f"모델 로드 완료: {MODEL_FILE} → MyModel")

    all_samples = build_sample_list_crop(TRAIN_CSV, DATA_ROOT)

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

        # ── WeightedRandomSampler 생성 (이 fold의 train_samples 기준) ──
        sampler = make_sampler(train_samples, mode=SAMPLER_MODE)

        train_ds = TBCTDatasetCrop(train_samples, mean, std, training=True)
        val_ds   = TBCTDatasetCrop(val_samples,   mean, std, training=False)

        # sampler 사용 시 shuffle=True와 동시 사용 불가 → shuffle 제거
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=2, pin_memory=False, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=False,
        )

        model = MyModel().to(DEVICE)

        # ── head별 독립 criterion (class 분포는 원본 기준으로 계산) ──
        print("Loss: head별 FocalLoss(gamma=2.0) × 2 heads (masked)")
        criteria = []
        for t in range(NUM_TASKS):
            labels_t = [s["labels"][t] for s in train_samples if s["labels"][t] >= 0]
            cnt_t = np.bincount(labels_t, minlength=2)
            if USE_FOCAL_LOSS:
                cw_t = torch.tensor(1.0 / np.maximum(cnt_t, 1), dtype=torch.float32)
                cw_t = cw_t / cw_t.sum() * len(cw_t)
                criteria.append(FocalLoss(alpha=cw_t.to(DEVICE), gamma=2.0, label_smoothing=0.05))
            else:
                criteria.append(nn.CrossEntropyLoss(label_smoothing=0.05))
            print(f"  head[{t}] {TASK_NAMES[t]}: Normal={cnt_t[0]}, Abnormal={cnt_t[1]}")

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        use_amp = torch.cuda.is_available()
        scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_f1    = -1.0
        no_improve = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            all_preds_flat, all_labels_flat = [], []

            print(f"\n[Fold {fold_idx+1} Epoch {epoch}/{NUM_EPOCHS}]  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            pbar = tqdm(train_loader, unit="batch")

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)                        # (B, 2, 2)
                    loss    = compute_multi_loss(outputs, labels, criteria)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

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

            # ── Validation: argmax(0.5) 기준 F1 → EarlyStopping ──
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

            print(f"Train Loss {train_loss:.4f} | F1(train) {train_f1:.4f}  [Acc {train_acc:.4f}]")
            print(f"Val   F1(macro) {f1_e:.4f} @ argmax  [Acc {val_acc:.4f}]")

            if f1_e > best_f1:
                best_f1    = f1_e
                no_improve = 0
                torch.save(model.state_dict(), fold_model_path)
                print(f"  >> Best F1 갱신: {best_f1:.4f} → {fold_model_path} 저장")
            else:
                no_improve += 1
                print(f"  (no improvement {no_improve}/{PATIENCE})")
                if no_improve >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"\n[Fold {fold_idx+1}] 완료. Best Val F1 = {best_f1:.4f}")

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

    # ── 체크포인트 덮어쓰기 경고 ──
    if os.path.exists(fullfit_path):
        print(f"\n[경고] {fullfit_path} 가 이미 존재합니다. 덮어쓰기됩니다.\n")

    print(f"사용 디바이스: {DEVICE}")
    print(f"사용 모델 파일: {MODEL_FILE}")
    print(f"CROP_SIDE_TO_RL: '{CROP_SIDE_TO_RL}'")
    print(f"Sampler: mode={SAMPLER_MODE!r}, RT_POS_TARGET_RATIO={RT_POS_TARGET_RATIO}")
    print(f"저장될 파일: {fullfit_path}, {preproc_stats_path}")
    print(f"학습 에폭: {fullfit_epochs}  (--fullfit-epochs로 변경 가능)")
    print(f"옵션: CLAHE={USE_CLAHE}, 2.5D={USE_25D}, FocalLoss={USE_FOCAL_LOSS}, "
          f"drop_last=True, NUM_TASKS={NUM_TASKS}(crop, 2-head + WeightedSampler)")

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

    # ── WeightedRandomSampler 생성 (전체 all_samples 기준) ──
    sampler = make_sampler(all_samples, mode=SAMPLER_MODE)

    train_ds = TBCTDatasetCrop(all_samples, mean, std, training=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=False, drop_last=True,
    )

    model = MyModel().to(DEVICE)

    print("Loss: head별 FocalLoss(gamma=2.0) × 2 heads (masked)")
    criteria = []
    for t in range(NUM_TASKS):
        labels_t = [s["labels"][t] for s in all_samples if s["labels"][t] >= 0]
        cnt_t = np.bincount(labels_t, minlength=2)
        if USE_FOCAL_LOSS:
            cw_t = torch.tensor(1.0 / np.maximum(cnt_t, 1), dtype=torch.float32)
            cw_t = cw_t / cw_t.sum() * len(cw_t)
            criteria.append(FocalLoss(alpha=cw_t.to(DEVICE), gamma=2.0, label_smoothing=0.05))
        else:
            criteria.append(nn.CrossEntropyLoss(label_smoothing=0.05))
        print(f"  head[{t}] {TASK_NAMES[t]}: Normal={cnt_t[0]}, Abnormal={cnt_t[1]}")

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

        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss    = compute_multi_loss(outputs, labels, criteria)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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

    torch.save(model.state_dict(), fullfit_path)
    print(f"\nFullfit 완료. 저장: {fullfit_path}  ({fullfit_epochs}에폭)")


# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TBCT Crop 2-head + WeightedRandomSampler 학습 스크립트 "
                    "(model_convNeXt-Tiny_crop_2head.py 전용)",
        usage="python3 train_crop_2head_sampler.py [--fullfit] [--fullfit-epochs N] [--dump-crops]",
    )
    parser.add_argument("--fullfit", action="store_true",
                        help="전체 데이터로 단일 모델 학습 (val/EarlyStopping 없음)")
    parser.add_argument("--fullfit-epochs", type=int, default=NUM_EPOCHS,
                        help=f"fullfit 학습 에폭 수 (기본={NUM_EPOCHS})")
    parser.add_argument("--dump-crops", action="store_true",
                        help="crop 디버그 이미지를 ./crop_2head_sampler_debug/에 저장 후 종료")

    args = parser.parse_args()

    DATA_ROOT = "./data" if os.path.exists("./data") else "../data"
    TRAIN_CSV = "train_set.csv" if os.path.exists("train_set.csv") else "../train_set.csv"

    if args.dump_crops:
        dump_crops(DATA_ROOT, TRAIN_CSV)
    elif args.fullfit:
        train_fullfit(args.fullfit_epochs)
    else:
        train()
