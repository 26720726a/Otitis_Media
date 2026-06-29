import os
import glob
import json
import argparse
import importlib.util
import re
import sys
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm


# ==========================================================
# [추론 설정]
# ==========================================================
DEFAULT_IMAGE_SIZE = 224
USE_TTA            = True
USE_SLICE_SMOOTH   = True
SMOOTH_WINDOW      = 3

NUM_TASKS  = 4
TASK_NAMES = ["rt_temporal", "lt_temporal", "rt_otitis", "lt_otitis"]


# ==========================================================
# [사용자 수동 threshold 설정]
#
# 여기 값만 바꾸면 됩니다.
###### 4개 head를 같은 값으로 쓰고 싶으면 전부 같은 값으로 설정하세요.
# ==========================================================
MANUAL_THRESHOLDS = {
    "rt_temporal": 0.7,
    "lt_temporal": 0.7,
    "rt_otitis":   0.7,
    "lt_otitis":   0.7,
}


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
# [파일 경로 자동 생성]
# ==========================================================
def make_paths_from_best_model(best_model_path):
    if not best_model_path.endswith(".pth"):
        raise ValueError("입력 파일은 반드시 .pth 파일이어야 합니다.")

    base = os.path.splitext(os.path.basename(best_model_path))[0]
    if not base.startswith("best_"):
        raise ValueError("가중치 파일명은 반드시 best_로 시작해야 합니다.")

    # fold 패턴: best_{model}_fold{k}.pth → model_name = {model}
    fold_match = re.match(r'^(best_.+?)_fold\d+$', base)
    # fullfit 패턴: best_{model}_fullfit.pth → model_name = {model}
    fullfit_match = re.match(r'^(best_.+?)_fullfit$', base)

    if fold_match:
        model_name = fold_match.group(1)[len("best_"):]
    elif fullfit_match:
        model_name = fullfit_match.group(1)[len("best_"):]
    else:
        model_name = base[len("best_"):]

    model_file = f"{model_name}.py"
    stats_path = f"preproc_{model_name}_stats.json"

    return model_name, model_file, best_model_path, stats_path


# ==========================================================
# [모델 파일 동적 로드]
# ==========================================================
def load_model_class(model_file):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file}")

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
# [전처리 통계 로드]
# ==========================================================
def load_aux(stats_path):
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        print(f"불러온 stats: mean={stats['mean']}, std={stats['std']}")
    else:
        print(f"[경고] 전처리 통계 파일 없음 → ImageNet 기본값 사용")
        stats = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "use_clahe": True, "use_25d": True,
            "image_size": DEFAULT_IMAGE_SIZE,
        }

    return stats



# ==========================================================
# [이미지 전처리]
# ==========================================================
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_gray(path, use_clahe):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if use_clahe:
        img = _clahe.apply(img)
    return img


def load_25d(img_dir, n, use_clahe, use_25d):
    cur_path = os.path.join(img_dir, f"{n:04d}.png")
    if not os.path.exists(cur_path):
        return None

    cur = load_gray(cur_path, use_clahe)
    if cur is None:
        return None

    if not use_25d:
        return np.stack([cur, cur, cur], axis=-1)

    prev_path = os.path.join(img_dir, f"{n-1:04d}.png") if n > 1 else None
    next_path = os.path.join(img_dir, f"{n+1:04d}.png") if n < 132 else None

    prev = load_gray(prev_path, use_clahe) if (prev_path and os.path.exists(prev_path)) else cur
    nxt  = load_gray(next_path, use_clahe) if (next_path and os.path.exists(next_path)) else cur

    if prev.shape != cur.shape:
        prev = cv2.resize(prev, (cur.shape[1], cur.shape[0]))
    if nxt.shape != cur.shape:
        nxt = cv2.resize(nxt, (cur.shape[1], cur.shape[0]))

    return np.stack([prev, cur, nxt], axis=-1)


def normalize_to_tensor(img_3ch, image_size, mean, std):
    img = cv2.resize(img_3ch, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    m = np.array(mean, dtype=np.float32)
    s = np.array(std,  dtype=np.float32)
    img = (img - m) / s
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.copy())


def make_tta_views(img_3ch, image_size, mean, std):
    views = [normalize_to_tensor(img_3ch, image_size, mean, std)]
    if not USE_TTA:
        return torch.stack(views, dim=0)

    h, w = img_3ch.shape[:2]
    for angle in (+5, -5):
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rot = cv2.warpAffine(img_3ch, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        views.append(normalize_to_tensor(rot, image_size, mean, std))

    zh, zw = int(h * 1.06), int(w * 1.06)
    big = cv2.resize(img_3ch, (zw, zh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (zh - h) // 2, (zw - w) // 2
    crop = big[y0:y0+h, x0:x0+w]
    views.append(normalize_to_tensor(crop, image_size, mean, std))

    return torch.stack(views, dim=0)


def smooth_1d(values, window=3):
    if window <= 1:
        return values
    k = window // 2
    out = [None] * len(values)
    for i, v in enumerate(values):
        if v is None:
            continue
        neigh = [values[j] for j in range(max(0, i-k), min(len(values), i+k+1))
                 if values[j] is not None]
        out[i] = sum(neigh) / len(neigh)
    return out


# ==========================================================
# [환자 1명 전체 슬라이스 예측 — 4 head 동시, 폴드 앙상블]
# ==========================================================
@torch.no_grad()
def predict_patient_slices(models, img_dir, slice_cols, device,
                           image_size, mean, std, use_clahe, use_25d):
    """
    models: list of models (K fold models, 단일 모델이면 길이 1)
    Returns: dict[slice_n] → np.array of shape (4,) = 폴드 평균 prob
    """
    results = {}

    for col in slice_cols:
        n = int(col)
        img = load_25d(img_dir, n, use_clahe, use_25d)
        if img is None:
            continue

        batch = make_tta_views(img, image_size, mean, std).to(device)  # (V, 3, H, W)

        fold_probs = []
        for m in models:
            outputs = m(batch)                                # (V, 4, 2)
            probs = torch.softmax(outputs, dim=2)[:, :, 1]   # (V, 4)
            fold_probs.append(probs.mean(dim=0))              # (4,)

        avg_probs = torch.stack(fold_probs, dim=0).mean(dim=0).cpu().numpy()  # (4,)
        results[n] = avg_probs

    return results


# ==========================================================
# [추론 메인]
# ==========================================================
def run_inference(best_model_path):
    DATA_ROOT = "./data"
    if not os.path.exists(DATA_ROOT):
        DATA_ROOT = "../data"

    TEMPLATE_PATH = "submission_template.csv"
    if not os.path.exists(TEMPLATE_PATH):
        TEMPLATE_PATH = "../submission_template.csv"

    OUTPUT_PATH = "submission_validation.csv"

    model_name, model_file, model_path, stats_path = \
        make_paths_from_best_model(best_model_path)

    print(f"모델 파일: {model_file} | 가중치: {model_path}")

    if not os.path.exists(model_file):
        print(f"\n[오류] 모델 구조 파일 없음: {model_file}\n")
        return
    if not os.path.exists(TEMPLATE_PATH):
        print(f"\n[오류] submission_template.csv 없음\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    MyModel = load_model_class(model_file)
    stats = load_aux(stats_path)

    thresholds = {name: float(MANUAL_THRESHOLDS.get(name, 0.5)) for name in TASK_NAMES}
    print("적용 Thresholds (MANUAL_THRESHOLDS):")
    for name in TASK_NAMES:
        print(f"  - {name}: {thresholds[name]:.4f}")

    image_size = stats.get("image_size", DEFAULT_IMAGE_SIZE)
    mean       = stats["mean"]
    std        = stats["std"]
    use_clahe  = stats["use_clahe"]
    use_25d    = stats["use_25d"]

    # ── 폴드/단일 가중치 로드 (소프트 보팅 앙상블) ──
    fold_paths = sorted(glob.glob(f"best_{model_name}_fold*.pth"))
    if not fold_paths:
        if not os.path.exists(model_path):
            print(f"\n[오류] 가중치 파일 없음: {model_path} (fold 파일도 없음)\n")
            return
        fold_paths = [model_path]
        print(f"단일 모델 사용: {model_path}")
    else:
        print(f"폴드 모델 {len(fold_paths)}개 감지")

    models = []
    for fp in fold_paths:
        m = MyModel().to(device)
        try:
            sd = torch.load(fp, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(fp, map_location=device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
        print(f"  로드: {fp}")

    df = pd.read_csv(TEMPLATE_PATH)
    slice_cols = [str(i) for i in range(1, 133)]

    print(f"추론 시작 (TTA={USE_TTA}, smooth={USE_SLICE_SMOOTH}, "
          f"{len(models)}모델 앙상블, head별 thresholds 사용)")

    # ── 환자별 예측 캐시 (같은 이미지 중복 예측 방지) ──
    prediction_cache = {}   # patient_id → dict[slice_n → (4,) probs]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        p_id     = str(int(row["No"]))
        task_idx = get_task_index(row["R/L"], row["Image number"])

        # 환자 최초 등장 시에만 전체 슬라이스 예측
        if p_id not in prediction_cache:
            p_dir = os.path.join(DATA_ROOT, p_id)
            if not os.path.exists(p_dir):
                p_dir = os.path.join(DATA_ROOT, "val", p_id)
            img_dir = os.path.join(p_dir, "PNG")

            if os.path.exists(img_dir):
                prediction_cache[p_id] = predict_patient_slices(
                    models, img_dir, slice_cols, device,
                    image_size, mean, std, use_clahe, use_25d
                )
            else:
                prediction_cache[p_id] = {}

        patient_preds = prediction_cache[p_id]

        # 해당 태스크의 prob만 추출
        task_probs = []
        for col in slice_cols:
            n = int(col)
            if n in patient_preds:
                task_probs.append(patient_preds[n][task_idx])
            else:
                task_probs.append(None)

        # 슬라이스 방향 smoothing
        if USE_SLICE_SMOOTH:
            task_probs = smooth_1d(task_probs, SMOOTH_WINDOW)

        # 해당 head의 threshold 적용 후 CSV에 기입
        task_name = TASK_NAMES[task_idx]
        threshold = thresholds.get(task_name, 0.5)

        for col, p in zip(slice_cols, task_probs):
            if p is None:
                continue
            df.at[idx, col] = int(1 if p >= threshold else 0)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n추론 완료: '{OUTPUT_PATH}' 생성")


# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TBCT Multi-Head 추론 스크립트",
        usage="python3 inference.py <best_model_file.pth>"
    )
    parser.add_argument("best_model_path", type=str,
                        help="예: best_model_densenet.pth")

    if len(sys.argv) < 2:
        print("\n  사용 예: python3 inference.py best_model_densenet.pth\n")
        sys.exit(1)

    args = parser.parse_args()
    run_inference(args.best_model_path)