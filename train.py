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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from tqdm import tqdm


# ==========================================================
# [하이퍼파라미터]
# ==========================================================
IMAGE_SIZE     = 224
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-4
NUM_EPOCHS     = 20
VAL_RATIO      = 0.20
PATIENCE       = 6
NUM_FOLDS      = 5
SEED           = 42

USE_CLAHE      = True
USE_25D        = True
USE_FOCAL_LOSS = True

NUM_TASKS      = 4
TASK_NAMES     = ["rt_temporal", "lt_temporal", "rt_otitis", "lt_otitis"]
IGNORE_LABEL   = -1   # 해당 태스크에 라벨이 없는 경우

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
        raise AttributeError(
            f"{model_file} 안에 MyModel 클래스가 없습니다. "
            f"반드시 class MyModel(nn.Module): 형태로 정의해야 합니다."
        )
    return model_module.MyModel


# ==========================================================
# [태스크 인덱스 판별]
# ==========================================================
def get_task_index(rl_value, image_number_value):
    rl  = str(rl_value).strip().lower()
    img = str(image_number_value).strip().lower()

    is_rt       = "rt" in rl
    is_temporal = "temporal" in img

    if is_rt and is_temporal:         return 0
    if (not is_rt) and is_temporal:   return 1
    if is_rt and (not is_temporal):   return 2
    return 3


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


def compute_dataset_stats(samples, max_n=2000):
    rng = random.Random(SEED)
    pick = samples if len(samples) <= max_n else rng.sample(samples, max_n)
    means, stds = [], []

    for s in tqdm(pick, desc="dataset stats"):
        x = load_25d_or_repeat(s["path"], s.get("prev"), s.get("next"))
        if x is None:
            continue
        x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        means.append(x.mean(axis=(0, 1)))
        stds.append(x.std(axis=(0, 1)))

    if len(means) == 0:
        raise RuntimeError("mean/std 계산에 사용할 이미지가 없습니다.")

    mean = np.mean(means, axis=0).tolist()
    std  = np.mean(stds, axis=0).tolist()
    std  = [max(s, 1e-3) for s in std]
    return mean, std


# ==========================================================
# [CSV 중복 행 제거]
# ==========================================================
def dedupe_rows(df):
    original_n = len(df)
    df_work = df.copy()
    first_col  = df_work.columns[0]
    second_col = df_work.columns[1]

    keep_indices = []
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
# [샘플 인덱싱: CSV → Multi-Task sample list]
#
#   핵심 변경:
#   같은 환자의 같은 슬라이스를 하나의 샘플로 묶고,
#   4개 태스크 라벨을 배열로 저장함.
#   라벨이 없는 태스크는 IGNORE_LABEL(-1)로 표시.
# ==========================================================
def build_sample_list(csv_path, base_path):
    df = pd.read_csv(csv_path)
    print(f"데이터 로드: {csv_path} ({len(df)}행)")
    df = dedupe_rows(df)

    rl_col  = "R/L"
    img_col = "Image number"
    slice_cols = [str(i) for i in range(1, 133)]

    # (patient_id, slice_n) → sample dict
    sample_dict = {}

    print("Multi-Task 샘플 인덱싱 중...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        p_id = str(int(row["No"]))
        task_idx = get_task_index(row[rl_col], row[img_col])

        # 이미지 디렉토리 탐색
        p_dir = os.path.join(base_path, p_id)
        if not os.path.exists(p_dir):
            p_dir = os.path.join(base_path, "train", p_id)
        img_dir = os.path.join(p_dir, "PNG")
        if not os.path.exists(img_dir):
            continue

        for col in slice_cols:
            if col not in row.index or pd.isna(row[col]):
                continue

            label = int(row[col])
            n = int(col)
            cur = os.path.join(img_dir, f"{n:04d}.png")
            if not os.path.exists(cur):
                continue

            key = (p_id, n)

            if key not in sample_dict:
                prev = os.path.join(img_dir, f"{n-1:04d}.png") if n > 1 else None
                nxt  = os.path.join(img_dir, f"{n+1:04d}.png") if n < 132 else None
                sample_dict[key] = {
                    "path":   cur,
                    "prev":   prev,
                    "next":   nxt,
                    "labels": [IGNORE_LABEL] * NUM_TASKS,
                    "pid":    p_id,
                }

            sample_dict[key]["labels"][task_idx] = label

    samples = list(sample_dict.values())
    print(f"총 샘플 수: {len(samples)} (이미지 기준, 각 샘플에 최대 4개 태스크 라벨)")

    if len(samples) == 0:
        raise RuntimeError("생성된 학습 샘플이 없습니다.")

    # 태스크별 라벨 분포 출력
    for t in range(NUM_TASKS):
        labels_t = [s["labels"][t] for s in samples if s["labels"][t] >= 0]
        if labels_t:
            cnt = np.bincount(labels_t, minlength=2)
            print(f"  {TASK_NAMES[t]}: Normal={cnt[0]}, Abnormal={cnt[1]}")
        else:
            print(f"  {TASK_NAMES[t]}: 라벨 없음")

    return samples


# ==========================================================
# [Dataset - Multi-Task]
# ==========================================================
class TBCTDataset(Dataset):
    def __init__(self, samples, mean, std, training=False):
        self.samples  = samples
        self.training = training
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _augment(self, img_3ch):
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
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        if self.training:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))

        labels = torch.tensor(s["labels"], dtype=torch.long)  # (4,)

        return torch.from_numpy(img.copy()), labels


# ==========================================================
# [Split]
# ==========================================================
def patient_level_split(samples, val_ratio=0.20, seed=SEED):
    pids = sorted({s["pid"] for s in samples})
    if len(pids) < 2:
        raise RuntimeError("환자 ID가 2개 미만입니다.")

    train_pids, val_pids = train_test_split(pids, test_size=val_ratio, random_state=seed)
    train_pids = set(train_pids)
    val_pids   = set(val_pids)

    return (
        [s for s in samples if s["pid"] in train_pids],
        [s for s in samples if s["pid"] in val_pids],
    )


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
# [Multi-Head Loss 계산]
#
#   outputs: (B, 4, 2) — 모델 출력
#   labels:  (B, 4)    — 각 태스크 라벨 (-1이면 무시)
#
#   유효한 태스크만 loss 계산 후 평균
# ==========================================================
def compute_multi_loss(outputs, labels, criterion):
    total_loss  = 0.0
    valid_tasks = 0

    for t in range(NUM_TASKS):
        mask = labels[:, t] >= 0
        if mask.sum() == 0:
            continue

        task_logits = outputs[mask, t, :]   # (N_valid, 2)
        task_labels = labels[mask, t]       # (N_valid,)

        total_loss += criterion(task_logits, task_labels)
        valid_tasks += 1

    if valid_tasks == 0:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    return total_loss / valid_tasks


# ==========================================================
# [Validation: head별 softmax 확률 수집 (threshold 탐색용)]
# ==========================================================
@torch.no_grad()
def collect_preds_prob(model, loader):
    """태스크별 (prob_array, label_array) 반환. prob = softmax의 class-1 확률."""
    model.eval()
    task_probs  = [[] for _ in range(NUM_TASKS)]
    task_labels = [[] for _ in range(NUM_TASKS)]

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        outputs = model(x)                              # (B, 4, 2)
        probs = torch.softmax(outputs, dim=2)[:, :, 1] # (B, 4)

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
# [학습 메인 — K-Fold]
# ==========================================================
def train(model_file):
    DATA_ROOT = "./data"
    if not os.path.exists(DATA_ROOT):
        DATA_ROOT = "../data"

    TRAIN_CSV = "train_set.csv"
    if not os.path.exists(TRAIN_CSV):
        TRAIN_CSV = "../train_set.csv"

    model_name         = os.path.splitext(os.path.basename(model_file))[0]
    preproc_stats_path = f"preproc_{model_name}_stats.json"

    print(f"사용 디바이스: {DEVICE}")
    print(f"사용 모델 파일: {model_file}")
    print(f"저장될 파일: best_{model_name}_fold{{k}}.pth (k=0..{NUM_FOLDS-1}), "
          f"{preproc_stats_path}")
    print(
        f"옵션: CLAHE={USE_CLAHE}, 2.5D={USE_25D}, "
        f"FocalLoss={USE_FOCAL_LOSS}, NUM_FOLDS={NUM_FOLDS}, "
        f"Multi-Head={NUM_TASKS}tasks"
    )

    MyModel = load_model_class(model_file)
    print(f"모델 로드 완료: {model_file} → MyModel")

    # ── 데이터 준비 (dedup은 build_sample_list 내에서 수행) ──
    all_samples = build_sample_list(TRAIN_CSV, DATA_ROOT)

    # ── mean/std: 전체 샘플 기준 1회 계산 (A2) ──
    print("데이터셋 mean/std 계산 중 (전체 샘플)...")
    mean, std = compute_dataset_stats(all_samples)
    print(f"  mean={mean}")
    print(f"  std ={std}")

    with open(preproc_stats_path, "w") as f:
        json.dump({
            "mean": mean, "std": std,
            "use_clahe": USE_CLAHE, "use_25d": USE_25D,
            "image_size": IMAGE_SIZE, "model_file": model_file,
            "num_tasks": NUM_TASKS,
        }, f, indent=2)

    # ── StratifiedGroupKFold 준비 (A1) ──
    # 환자별 양성 라벨 수집 → 비율 → qcut 3구간 bin
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

    kfold       = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(kfold.split(all_samples, sample_bin, groups))

    # ── K-Fold 학습 루프 ──
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"[Fold {fold_idx+1}/{NUM_FOLDS}]")
        print(f"{'='*60}")

        fold_model_path = f"best_{model_name}_fold{fold_idx}.pth"

        train_samples = [all_samples[i] for i in train_idx]
        val_samples   = [all_samples[i] for i in val_idx]
        print(f"Train 샘플: {len(train_samples)} | Val 샘플: {len(val_samples)}")

        # 폴드별 클래스 분포 (FocalLoss 가중치용)
        fold_train_labels = [l for s in train_samples for l in s["labels"] if l >= 0]
        cls_count = np.bincount(fold_train_labels, minlength=2)
        pos_ratio = cls_count[1] / max(cls_count.sum(), 1)
        print(f"Train 클래스 분포: Normal={cls_count[0]}, Abnormal={cls_count[1]} "
              f"(병변 비율 {pos_ratio*100:.1f}%)")

        # ── DataLoader ──
        train_ds = TBCTDataset(train_samples, mean, std, training=True)
        val_ds   = TBCTDataset(val_samples,   mean, std, training=False)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=False, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=False
        )

        # ── 모델/loss/optimizer (폴드마다 fresh) ──
        model = MyModel().to(DEVICE)

        if USE_FOCAL_LOSS:
            cw = torch.tensor(1.0 / np.maximum(cls_count, 1), dtype=torch.float32)
            cw = cw / cw.sum() * len(cw)
            criterion = FocalLoss(alpha=cw.to(DEVICE), gamma=2.0, label_smoothing=0.05)
            print("Loss: FocalLoss(gamma=2.0) × 4 heads (masked)")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            print("Loss: CrossEntropyLoss × 4 heads (masked)")

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

        use_amp = torch.cuda.is_available()
        scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_f1    = -1.0
        no_improve = 0

        # ── 에폭 루프 ──
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
                    outputs = model(inputs)                      # (B, 4, 2)
                    loss = compute_multi_loss(outputs, labels, criterion)

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
            train_f1   = f1_score(all_labels_flat, all_preds_flat, average="macro", zero_division=0)

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
                best_f1 = f1_e
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
# [Full-fit 단일 모델 학습 — 전체 데이터, EarlyStopping 없음]
# ==========================================================
def train_fullfit(model_file, fullfit_epochs):
    DATA_ROOT = "./data"
    if not os.path.exists(DATA_ROOT):
        DATA_ROOT = "../data"
    TRAIN_CSV = "train_set.csv"
    if not os.path.exists(TRAIN_CSV):
        TRAIN_CSV = "../train_set.csv"

    model_name         = os.path.splitext(os.path.basename(model_file))[0]
    fullfit_path       = f"best_{model_name}_fullfit.pth"
    preproc_stats_path = f"preproc_{model_name}_stats.json"

    print(f"사용 디바이스: {DEVICE}")
    print(f"사용 모델 파일: {model_file}")
    print(f"저장될 파일: {fullfit_path}, {preproc_stats_path}")
    print(f"학습 에폭: {fullfit_epochs}  (--fullfit-epochs로 변경 가능)")
    print(
        f"옵션: CLAHE={USE_CLAHE}, 2.5D={USE_25D}, "
        f"FocalLoss={USE_FOCAL_LOSS}, drop_last=True, "
        f"Multi-Head={NUM_TASKS}tasks"
    )

    MyModel = load_model_class(model_file)

    # ── 전체 데이터 로드 ──
    all_samples = build_sample_list(TRAIN_CSV, DATA_ROOT)
    print(f"전체 샘플: {len(all_samples)}")

    # ── mean/std: 전체 샘플 계산 → stats.json 갱신 ──
    print("데이터셋 mean/std 계산 중 (전체 샘플)...")
    mean, std = compute_dataset_stats(all_samples)
    print(f"  mean={mean}")
    print(f"  std ={std}")
    with open(preproc_stats_path, "w") as f:
        json.dump({
            "mean": mean, "std": std,
            "use_clahe": USE_CLAHE, "use_25d": USE_25D,
            "image_size": IMAGE_SIZE, "model_file": model_file,
            "num_tasks": NUM_TASKS,
        }, f, indent=2)

    # ── 클래스 분포 (FocalLoss 가중치) ──
    all_train_labels = [l for s in all_samples for l in s["labels"] if l >= 0]
    cls_count = np.bincount(all_train_labels, minlength=2)
    pos_ratio = cls_count[1] / max(cls_count.sum(), 1)
    print(f"클래스 분포: Normal={cls_count[0]}, Abnormal={cls_count[1]} "
          f"(병변 비율 {pos_ratio*100:.1f}%)")

    # ── DataLoader (drop_last=True: BatchNorm 배치=1 방지) ──
    train_ds = TBCTDataset(all_samples, mean, std, training=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=False, drop_last=True
    )

    # ── 모델/loss/optimizer ──
    model = MyModel().to(DEVICE)

    if USE_FOCAL_LOSS:
        cw = torch.tensor(1.0 / np.maximum(cls_count, 1), dtype=torch.float32)
        cw = cw / cw.sum() * len(cw)
        criterion = FocalLoss(alpha=cw.to(DEVICE), gamma=2.0, label_smoothing=0.05)
        print("Loss: FocalLoss(gamma=2.0) × 4 heads (masked)")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=fullfit_epochs, eta_min=1e-6
    )

    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── 학습 루프 (val / EarlyStopping 없음) ──
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
                loss = compute_multi_loss(outputs, labels, criterion)

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
        train_f1   = f1_score(all_labels_flat, all_preds_flat, average="macro", zero_division=0)
        train_acc  = accuracy_score(all_labels_flat, all_preds_flat)
        print(f"Train Loss {train_loss:.4f} | F1(train) {train_f1:.4f}  [Acc {train_acc:.4f}]")

    torch.save(model.state_dict(), fullfit_path)
    print(f"\nFullfit 완료. 저장: {fullfit_path}  ({fullfit_epochs}에폭)")
    print(f"추론: python3 inference.py {fullfit_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TBCT Multi-Head 학습 스크립트",
        usage="python3 train.py <model_file.py> [--fullfit] [--fullfit-epochs N]"
    )
    parser.add_argument("model_file", type=str,
                        help="학습에 사용할 모델 파일 경로. 예: model_densenet.py")
    parser.add_argument("--fullfit", action="store_true",
                        help="전체 데이터로 단일 모델 학습 (val/EarlyStopping 없음)")
    parser.add_argument("--fullfit-epochs", type=int, default=NUM_EPOCHS,
                        help=f"fullfit 학습 에폭 수 (기본={NUM_EPOCHS}, 폴드 평균 best epoch 권장)")

    if len(sys.argv) < 2:
        print("\n[오류] 모델 파일명을 입력해야 합니다.")
        print("  K-Fold 학습:   python3 train.py model_densenet.py")
        print("  Full-fit 학습: python3 train.py model_densenet.py --fullfit --fullfit-epochs 12\n")
        sys.exit(1)

    args = parser.parse_args()
    if args.fullfit:
        train_fullfit(args.model_file, args.fullfit_epochs)
    else:
        train(args.model_file)