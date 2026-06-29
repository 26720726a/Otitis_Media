"""
threshold 스윕 스크립트 (model_convNeXt-Tiny_crop_2head 전용)

목적:
    추론(모델이 모든 이미지를 보는 비싼 과정)을 딱 1번만 돌려 슬라이스별 확률을
    캐시한 뒤, threshold만 여러 개 갈아끼우며 채점한다. 재추론 0번.

    - 각 4-head 태스크(rt_temporal/lt_temporal/rt_otitis/lt_otitis)별로
      threshold → macroF1 표를 출력 (어느 값이 최적인지 눈으로 확인).
    - 태스크별 최적 threshold를 적용했을 때의 "전체 합산 F1"(= evaluation.py와
      동일한 실제 대회 점수)을 baseline(전부 0.50)과 비교해서 출력.

    추론 로직은 inference_crop_2head.py의 함수를 그대로 import해서 사용하므로
    실제 추론과 100% 동일한 확률/스무딩이 쓰인다 (분포 어긋남 없음).
    TASK_TO_CROP도 inference_crop_2head.py에서 그대로 가져오므로
    rt_otitis·lt_otitis가 모두 otitis head(1)을 가리킴이 자동 보장된다.

사용법:
    python3 sweep_thresholds_2head.py best_model_convNeXt-Tiny_crop_2head_fold0.pth

    ※ val_set.csv (정답 라벨)와 data/ 디렉토리가 있어야 함.
    ※ inference_crop_2head.py 가 같은 폴더에 있어야 함.

★ 단일 fold 결과에 맞춘 threshold는 val에 과적합될 수 있다.
  여기서는 "대략의 방향"만 잡고, 최종 확정은 full 5-fold 앙상블에서 할 것.
"""

import os
import sys
import glob
import importlib.util

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


# ==========================================================
# [설정]
# ==========================================================
INFER_FILE = "inference_crop_2head.py"      # 추론 로직을 가져올 파일
VAL_CSV    = "val_set.csv"                   # 정답 라벨
GRID       = np.round(np.arange(0.30, 0.71, 0.05), 2)  # 시험할 threshold 후보

SLICE_COLS = [str(i) for i in range(1, 133)]
TASKS      = ["rt_temporal", "lt_temporal", "rt_otitis", "lt_otitis"]


# ==========================================================
# [inference_crop_2head.py 동적 import]
#   (__main__ 가드 덕분에 run_inference는 자동 실행되지 않음)
# ==========================================================
def load_infer_module(path):
    if not os.path.exists(path):
        print(f"[오류] {path} 를 찾을 수 없습니다. 같은 폴더에 두세요.")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("infer2", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==========================================================
# [모델/스탯 로드 — run_inference와 동일한 절차]
# ==========================================================
def load_models_and_stats(infer, best_model_path, device):
    model_name, model_file, model_path, stats_path = \
        infer.make_paths_from_best_model(best_model_path)

    MyModel = infer.load_model_class(model_file)
    stats   = infer.load_aux(stats_path)

    fold_paths = sorted(glob.glob(f"best_{model_name}_fold*.pth"))
    if not fold_paths:
        fold_paths = [model_path]
        print(f"단일 모델 사용: {model_path}")
    else:
        print(f"폴드 모델 {len(fold_paths)}개 감지 (앙상블)")

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

    return models, stats


# ==========================================================
# [확률 캐시 구축 + (prob, true) 페어 수집]
#   추론은 여기서 딱 1번. 이후 threshold 스윕은 이 배열만 재사용.
# ==========================================================
def collect_probs(infer, models, stats, device):
    DATA_ROOT = "./data" if os.path.exists("./data") else "../data"

    image_size = stats.get("image_size", infer.DEFAULT_IMAGE_SIZE)
    mean, std  = stats["mean"], stats["std"]
    use_clahe  = stats["use_clahe"]
    use_25d    = stats["use_25d"]

    ans_df = pd.read_csv(VAL_CSV).sort_values(by="No").reset_index(drop=True)

    per_task = {t: {"prob": [], "true": []} for t in TASKS}
    prediction_cache = {}

    print(f"\n확률 계산 중 (추론 1회, {len(models)}모델 앙상블)...")
    from tqdm import tqdm
    for _, row in tqdm(ans_df.iterrows(), total=len(ans_df)):
        p_id      = str(int(row["No"]))
        task_idx  = infer.get_task_index(row["R/L"], row["Image number"])
        task_name = infer.OUTPUT_TASK_NAMES[task_idx]
        side, head_idx = infer.TASK_TO_CROP[task_idx]

        # 환자 최초 등장 시 전체 슬라이스 좌/우 예측 (비싼 단계, 1회만)
        if p_id not in prediction_cache:
            p_dir = os.path.join(DATA_ROOT, p_id)
            if not os.path.exists(p_dir):
                p_dir = os.path.join(DATA_ROOT, "val", p_id)
            # ★ train_crop_2head.py / inference_crop_2head.py와 동일하게 PNG_soft 사용
            img_dir = os.path.join(p_dir, "PNG_soft")
            if os.path.exists(img_dir):
                prediction_cache[p_id] = infer.predict_patient_slices_crop(
                    models, img_dir, SLICE_COLS, device,
                    image_size, mean, std, use_clahe, use_25d,
                )
            else:
                prediction_cache[p_id] = {}

        patient_preds = prediction_cache[p_id]

        # 해당 (side, head_idx) 확률 시퀀스 추출
        # 2-head 모델: head_idx는 0(temporal) 또는 1(otitis)만 존재
        seq = []
        for col in SLICE_COLS:
            n = int(col)
            seq.append(float(patient_preds[n][side][head_idx]) if n in patient_preds else None)

        # 실제 추론과 동일하게 슬라이스 방향 스무딩
        if infer.USE_SLICE_SMOOTH:
            seq = infer.smooth_1d(seq, infer.SMOOTH_WINDOW)

        # 정답 라벨과 페어링
        for col, p in zip(SLICE_COLS, seq):
            if p is None:
                continue
            v = row[col]
            if pd.isna(v):
                continue
            per_task[task_name]["prob"].append(p)
            per_task[task_name]["true"].append(int(v))

    for t in TASKS:
        per_task[t]["prob"] = np.array(per_task[t]["prob"])
        per_task[t]["true"] = np.array(per_task[t]["true"])
    return per_task


# ==========================================================
# [스윕 + 채점]
# ==========================================================
def macro_f1(true, prob, thr):
    pred = (prob >= thr).astype(int)
    return f1_score(true, pred, average="macro", labels=[0, 1], zero_division=0)


def overall_pooled_f1(per_task, thr_map):
    """evaluation.py와 동일: 모든 태스크 슬라이스를 합쳐 단일 macro F1."""
    at, ap = [], []
    for t in TASKS:
        true = per_task[t]["true"]
        pred = (per_task[t]["prob"] >= thr_map[t]).astype(int)
        at.extend(true)
        ap.extend(pred)
    return f1_score(at, ap, average="macro", labels=[0, 1], zero_division=0)


def run(best_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    infer = load_infer_module(INFER_FILE)
    if not os.path.exists(VAL_CSV):
        print(f"[오류] {VAL_CSV} 없음")
        return

    # ── TASK_TO_CROP 점검: rt_otitis·lt_otitis가 모두 head_idx=1인지 확인 ──
    print("\n[점검] infer.TASK_TO_CROP 확인 (2-head: head_idx >= 2가 없어야 함)")
    for task_idx, task_name in enumerate(infer.OUTPUT_TASK_NAMES):
        side, head_idx = infer.TASK_TO_CROP[task_idx]
        marker = " ✓" if head_idx <= 1 else " ★★ 오류: head_idx >= 2"
        print(f"  task_idx={task_idx} ({task_name:12s}) → side={side!r:7s}, head_idx={head_idx}{marker}")
    otitis_heads = {infer.TASK_TO_CROP[2][1], infer.TASK_TO_CROP[3][1]}
    if otitis_heads == {1}:
        print("  [OK] rt_otitis·lt_otitis 둘 다 head_idx=1 (공유 otitis head) ✓")
    else:
        print(f"  [경고] otitis head_idx 집합이 {{1}}이 아님: {otitis_heads}")

    models, stats = load_models_and_stats(infer, best_model_path, device)
    per_task = collect_probs(infer, models, stats, device)

    # ── 태스크별 threshold → macroF1 표 ──
    print("\n" + "=" * 70)
    print("태스크별 threshold 스윕 (각 칸 = 그 threshold일 때 macroF1)")
    print("=" * 70)
    header = "TASK".ljust(14) + "".join(f"{g:>7.2f}" for g in GRID)
    print(header)
    print("-" * 70)

    best_thr = {}
    for t in TASKS:
        true, prob = per_task[t]["true"], per_task[t]["prob"]
        if len(true) == 0:
            print(f"{t:<14}(데이터 없음)")
            best_thr[t] = 0.50
            continue
        scores = [macro_f1(true, prob, g) for g in GRID]
        best_i = int(np.argmax(scores))
        best_thr[t] = float(GRID[best_i])
        row = f"{t:<14}" + "".join(
            (f"{s:>7.4f}" if i != best_i else f"{('*'+format(s,'.3f')):>7}")
            for i, s in enumerate(scores)
        )
        print(row)
    print("-" * 70)
    print("* = 해당 태스크 최적 threshold")
    print("최적값:", {t: best_thr[t] for t in TASKS})

    # ── 태스크별 F1 (최적 threshold) ──
    print("\n" + "=" * 70)
    print("태스크별 최적 threshold에서의 F1")
    print("=" * 70)
    for t in TASKS:
        true, prob = per_task[t]["true"], per_task[t]["prob"]
        if len(true) == 0:
            print(f"  {t:<14}: 데이터 없음")
            continue
        f1 = macro_f1(true, prob, best_thr[t])
        prec = precision_score(true, (prob >= best_thr[t]).astype(int),
                               average="macro", labels=[0, 1], zero_division=0)
        rec  = recall_score(true,  (prob >= best_thr[t]).astype(int),
                            average="macro", labels=[0, 1], zero_division=0)
        print(f"  {t:<14}: thr={best_thr[t]:.2f}  F1={f1:.4f}  "
              f"Prec={prec:.4f}  Recall={rec:.4f}")

    # ── 전체 합산 F1 (실제 대회 점수) 비교 ──
    base_map = {t: 0.50 for t in TASKS}
    base_overall = overall_pooled_f1(per_task, base_map)
    best_overall = overall_pooled_f1(per_task, best_thr)

    print("\n" + "=" * 70)
    print("전체 합산 F1 (← evaluation.py와 동일 = 실제 점수)")
    print("=" * 70)
    print(f"  baseline (전부 0.50)         : {base_overall:.4f}")
    print(f"  태스크별 최적 적용            : {best_overall:.4f}")
    print("=" * 70)

    # ── inference_crop_2head.py에 바로 붙여넣을 수 있는 dict 출력 ──
    print("\n" + "=" * 70)
    print("inference_crop_2head.py의 MANUAL_THRESHOLDS에 붙여넣을 값:")
    print("=" * 70)
    print("MANUAL_THRESHOLDS = {")
    for t in TASKS:
        print(f'    "{t}": {best_thr[t]:.2f},')
    print("}")

    print("\n[주의] 이 threshold는 단일 fold val에 과적합될 수 있음.")
    print("       최종 확정은 full 5-fold 앙상블 결과에서 할 것.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 sweep_thresholds_2head.py "
              "best_model_convNeXt-Tiny_crop_2head_fold0.pth")
        sys.exit(1)
    run(sys.argv[1])
