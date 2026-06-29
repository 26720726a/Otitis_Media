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
USE_TTA            = False
USE_SLICE_SMOOTH   = True
SMOOTH_WINDOW      = 3

# crop 3-head 모델의 출력 태스크
CROP_TASK_NAMES = ["temporal", "rt_otitis", "lt_otitis"]   # head index 0, 1, 2

# 최종 submission CSV의 4-head 순서 (변경 금지)
OUTPUT_TASK_NAMES = ["rt_temporal", "lt_temporal", "rt_otitis", "lt_otitis"]


# ==========================================================
# [★ 좌우 crop 방향 설정 — train_crop_3head.py와 반드시 일치시킬 것]
#
# train_crop_3head.py의 CROP_SIDE_TO_RL과 이 값이 다르면
# 좌우 확률이 뒤바뀌어 점수가 무너집니다.
# 학습 시 사용한 값과 동일하게 설정하세요.
#
# "standard": 방사선 표준 규약
#   LEFT  crop → 환자 Rt → rt_temporal, rt_otitis
#   RIGHT crop → 환자 Lt → lt_temporal, lt_otitis
#
# "flipped": 반대 방향
#   LEFT  crop → 환자 Lt → lt_temporal, lt_otitis
#   RIGHT crop → 환자 Rt → rt_temporal, rt_otitis
# ==========================================================
CROP_SIDE_TO_RL = "standard"  # ★ train_crop_3head_3win.py와 동일하게 유지

# ==========================================================
# [3-window 입력 설정 — train_crop_3head_3win.py와 완전히 동일해야 함]
#
# ★ WIN_SUBDIRS 순서 = 채널 순서. train과 다르면 채널 분포가 어긋남.
# ==========================================================
USE_3WINDOW  = True
WIN_SUBDIRS  = ["PNG_bone", "PNG_soft", "PNG"]   # 채널0, 1, 2 순서 (변경 금지)


# ==========================================================
# [4-head task_idx → (crop_side, head_idx) 매핑]
#
# task_idx: 0=rt_temporal, 1=lt_temporal, 2=rt_otitis, 3=lt_otitis
# head_idx: 0=temporal, 1=rt_otitis, 2=lt_otitis
#
# standard 기준:
#   rt_temporal → LEFT  crop, temporal head(0)
#   lt_temporal → RIGHT crop, temporal head(0)
#   rt_otitis   → LEFT  crop, rt_otitis head(1)
#   lt_otitis   → RIGHT crop, lt_otitis head(2)
#
# ★ crop 기준 W//2 도 train_crop_3head.py와 동일. 변경 금지.
# ==========================================================
if CROP_SIDE_TO_RL == "standard":
    TASK_TO_CROP = {
        0: ("left",  0),   # rt_temporal → left  crop, temporal head (0)
        1: ("right", 0),   # lt_temporal → right crop, temporal head (0)
        2: ("left",  1),   # rt_otitis   → left  crop, rt_otitis head (1)
        3: ("right", 2),   # lt_otitis   → right crop, lt_otitis head (2)
    }
else:  # "flipped"
    TASK_TO_CROP = {
        0: ("right", 0),   # rt_temporal → right crop, temporal head (0)
        1: ("left",  0),   # lt_temporal → left  crop, temporal head (0)
        2: ("right", 1),   # rt_otitis   → right crop, rt_otitis head (1)
        3: ("left",  2),   # lt_otitis   → left  crop, lt_otitis head (2)
    }


# ==========================================================
# [사용자 수동 threshold 설정]
#
# 4개 태스크에 각각 독립 threshold를 설정합니다.
# 초기값은 모두 0.50. 실험 결과에 따라 조정하세요.
# ==========================================================
MANUAL_THRESHOLDS = {
    "rt_temporal": 0.55,
    "lt_temporal": 0.55,
    "rt_otitis":   0.55,
    "lt_otitis":   0.60,
}


# ==========================================================
# [4-head task index 판별 — submission 행 매핑용]
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

    fold_match   = re.match(r'^(best_.+?)_fold\d+$', base)
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
        # train_crop_3head_3win.py가 저장한 crop_side_to_rl 값을 검증
        saved_rl = stats.get("crop_side_to_rl")
        if saved_rl and saved_rl != CROP_SIDE_TO_RL:
            print(f"\n  [★ 경고] stats에 저장된 CROP_SIDE_TO_RL='{saved_rl}'이 "
                  f"현재 설정 '{CROP_SIDE_TO_RL}'와 다릅니다!")
            print(f"  학습과 추론의 방향이 불일치하면 좌우가 뒤바뀝니다.")
            print(f"  파일 상단의 CROP_SIDE_TO_RL을 '{saved_rl}'로 맞추세요.\n")
    else:
        print(f"[경고] 전처리 통계 파일 없음 ({stats_path}) → ImageNet 기본값 사용")
        stats = {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
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


# ==========================================================
# [3-window 로딩 — bone/soft/png 채널]
#
# ★ train_crop_3head_3win.py / inference_crop_3head_3win.py 양쪽에서
#   완전히 동일한 로직 사용. WIN_SUBDIRS 순서가 채널 순서.
# ==========================================================
def _swap_win_dir(cur_path, subdir):
    """cur_path의 폴더명(PNG)을 subdir(PNG_bone 등)로 교체한 경로 반환."""
    d       = os.path.dirname(cur_path)   # .../PNG (또는 다른 window 폴더)
    pid_dir = os.path.dirname(d)          # .../{pid}
    fname   = os.path.basename(cur_path)  # {n:04d}.png
    return os.path.join(pid_dir, subdir, fname)


def load_3window(cur_path, use_clahe):
    """3개 window 폴더에서 같은 슬라이스를 읽어 (H, W, 3) 배열로 반환.
    window 파일이 없으면 원본 PNG로 폴백. 원본도 없으면 None 반환.
    채널 순서: WIN_SUBDIRS = [bone, soft, png] → [채널0, 채널1, 채널2].
    ★ train_crop_3head_3win.py의 load_3window(cur_path)와 동일 로직.
       train에서는 USE_CLAHE 상수 참조, 여기서는 stats에서 받은 use_clahe 사용.
    """
    chans = []
    ref   = None   # 원본 PNG (폴백용 캐시)
    for sub in WIN_SUBDIRS:
        p = _swap_win_dir(cur_path, sub)
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            if ref is None:
                ref = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
            g = ref
        if g is None:
            return None   # cur_path 자체가 존재하지 않는 경우
        if use_clahe:
            g = _clahe.apply(g)
        chans.append(g)
    # 크기 불일치 대비 — 채널0 기준 resize
    h, w = chans[0].shape[:2]
    for i in range(1, 3):
        if chans[i].shape[:2] != (h, w):
            chans[i] = cv2.resize(chans[i], (w, h))
    return np.stack(chans, axis=-1)   # (H, W, 3) = [bone, soft, png]


# ==========================================================
# [Letterbox resize — 비율 유지 + 검정 패딩]
#
# ★ train_crop_3head.py / inference_crop_3head.py 양쪽에서 완전히 동일한 코드 사용.
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


def normalize_to_tensor(img_3ch, image_size, mean, std):
    img = cv2.resize(img_3ch, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    m = np.array(mean, dtype=np.float32)
    s = np.array(std,  dtype=np.float32)
    img = (img - m) / s
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.copy())


def make_tta_views(img_3ch, image_size, mean, std):
    """crop된 이미지에 TTA 적용.
    view 구성 (USE_TTA=True 시 4개):
      [0] 원본  [1] 회전 +5°  [2] 회전 -5°  [3] 줌 1.06
    ★ 좌우/상하 flip 없음 (R/L 진단 의미 보존).
    """
    views = [normalize_to_tensor(img_3ch, image_size, mean, std)]
    if not USE_TTA:
        return torch.stack(views, dim=0)

    h, w = img_3ch.shape[:2]

    for angle in (+5, -5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(img_3ch, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        views.append(normalize_to_tensor(rot, image_size, mean, std))

    zh, zw = int(h * 1.06), int(w * 1.06)
    big  = cv2.resize(img_3ch, (zw, zh), interpolation=cv2.INTER_LINEAR)
    y0   = (zh - h) // 2
    x0   = (zw - w) // 2
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
# [환자 1명 전체 슬라이스 예측 — crop 3-head 버전]
#
# 각 슬라이스에 대해:
#   ① full 2.5D 이미지 로드
#   ② 가로 W//2 기준으로 left / right crop 분리
#      ★ 이 W//2 기준이 train_crop_3head.py의 __getitem__과 동일해야 함
#   ③ 각 crop에 TTA + 폴드 앙상블 적용
#   ④ 결과: {"left": (3,), "right": (3,)} — 각 (temporal, rt_otitis, lt_otitis) 확률
#      head_idx 0=temporal, 1=rt_otitis, 2=lt_otitis
# ==========================================================
@torch.no_grad()
def predict_patient_slices_crop(models, img_dir, slice_cols, device,
                                image_size, mean, std, use_clahe, use_25d):
    results = {}   # slice_n → {"left": np.array(3,), "right": np.array(3,)}

    for col in slice_cols:
        n = int(col)
        if USE_3WINDOW:
            cur_path = os.path.join(img_dir, f"{n:04d}.png")
            if not os.path.exists(cur_path):
                continue
            full_img = load_3window(cur_path, use_clahe)
        else:
            full_img = load_25d(img_dir, n, use_clahe, use_25d)
        if full_img is None:
            continue

        h, w = full_img.shape[:2]

        # ★ crop 기준: W//2  (train_crop_3head_3win.py TBCTDatasetCrop.__getitem__과 동일)
        crops = {
            "left":  full_img[:, :w // 2],
            "right": full_img[:, w // 2:],
        }

        slice_result = {}
        for side, crop_img in crops.items():
            # TTA는 crop 이후 적용 (학습 시 증강 순서와 동일)
            batch = make_tta_views(crop_img, image_size, mean, std).to(device)  # (V, 3, H, W)

            fold_probs = []
            for m in models:
                outputs = m(batch)                                  # (V, 3, 2)
                probs   = torch.softmax(outputs, dim=2)[:, :, 1]   # (V, 3)
                fold_probs.append(probs.mean(dim=0))                # (3,)

            avg = torch.stack(fold_probs, dim=0).mean(dim=0).cpu().numpy()  # (3,)
            slice_result[side] = avg

        results[n] = slice_result

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
    print(f"CROP_SIDE_TO_RL: '{CROP_SIDE_TO_RL}'")

    if not os.path.exists(model_file):
        print(f"\n[오류] 모델 구조 파일 없음: {model_file}\n")
        return
    if not os.path.exists(TEMPLATE_PATH):
        print(f"\n[오류] submission_template.csv 없음\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    MyModel = load_model_class(model_file)
    stats   = load_aux(stats_path)   # CROP_SIDE_TO_RL 불일치 경고도 여기서 출력

    # threshold 출력 (4개 태스크 각각 어떤 값이 적용되는지)
    print("적용 Thresholds (MANUAL_THRESHOLDS):")
    for task_idx, task_name in enumerate(OUTPUT_TASK_NAMES):
        thr = float(MANUAL_THRESHOLDS.get(task_name, 0.5))
        print(f"  {task_name}: {thr:.4f}")

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

    print(f"\n추론 시작 (TTA={USE_TTA}, smooth={USE_SLICE_SMOOTH}, "
          f"{len(models)}모델 앙상블, crop 3-head 3-window 방식)")

    # ── 환자별 슬라이스 예측 캐시 ──
    # prediction_cache[pid][slice_n] = {"left": (3,), "right": (3,)}
    prediction_cache = {}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        p_id     = str(int(row["No"]))
        task_idx = get_task_index(row["R/L"], row["Image number"])
        side, head_idx = TASK_TO_CROP[task_idx]

        # 환자 최초 등장 시 전체 슬라이스 예측 (좌/우 동시)
        if p_id not in prediction_cache:
            p_dir = os.path.join(DATA_ROOT, p_id)
            if not os.path.exists(p_dir):
                p_dir = os.path.join(DATA_ROOT, "val", p_id)
            img_dir = os.path.join(p_dir, "PNG")

            if os.path.exists(img_dir):
                prediction_cache[p_id] = predict_patient_slices_crop(
                    models, img_dir, slice_cols, device,
                    image_size, mean, std, use_clahe, use_25d,
                )
            else:
                prediction_cache[p_id] = {}

        patient_preds = prediction_cache[p_id]

        # 해당 (side, head_idx)의 슬라이스별 확률 시퀀스 추출
        task_probs = []
        for col in slice_cols:
            n = int(col)
            if n in patient_preds:
                task_probs.append(float(patient_preds[n][side][head_idx]))
            else:
                task_probs.append(None)

        # 슬라이스 방향 smoothing
        if USE_SLICE_SMOOTH:
            task_probs = smooth_1d(task_probs, SMOOTH_WINDOW)

        # threshold 적용 후 CSV 기입 (4-head task 기준 개별 threshold)
        threshold = float(MANUAL_THRESHOLDS.get(OUTPUT_TASK_NAMES[task_idx], 0.5))

        for col, p in zip(slice_cols, task_probs):
            if p is None:
                continue
            df.at[idx, col] = int(1 if p >= threshold else 0)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n추론 완료: '{OUTPUT_PATH}' 생성")


# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TBCT Crop 3-head 3-window 입력 추론 스크립트 (model_convNeXt-Tiny_crop_3head_3win.py 전용)",
        usage="python3 inference_crop_3head_3win.py <best_model_crop_3head_3win_file.pth>",
    )
    parser.add_argument("best_model_path", type=str,
                        help="예: best_model_convNeXt-Tiny_crop_3head_3win_fold0.pth")

    if len(sys.argv) < 2:
        print("\n  사용 예: python3 inference_crop_3head_3win.py "
              "best_model_convNeXt-Tiny_crop_3head_3win_fold0.pth\n")
        sys.exit(1)

    args = parser.parse_args()
    run_inference(args.best_model_path)
