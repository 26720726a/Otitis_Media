"""
태스크별 상세 분석 스크립트 (원본 evaluation.py는 건드리지 않음)

기존 evaluation.py는 전체 슬라이스를 Normal/Otitis 2클래스로 합쳐서 채점한다.
이 스크립트는 같은 데이터를 4개 태스크
  - rt_temporal (우측 측두골)
  - lt_temporal (좌측 측두골)
  - rt_otitis   (우측 중이염)
  - lt_otitis   (좌측 중이염)
로 분리해서, 각 태스크별 F1/precision/recall과 Normal/Otitis 분해를 보여준다.

→ "전체 0.86" 뒤에 숨은 "어느 태스크가 약한지"를 드러내는 게 목적.

사용법:
    python3 evaluation_per_task.py submission_validation.csv
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def task_of(rl_value, image_number_value):
    """R/L, Image number → 4개 태스크 중 하나로 분류."""
    rl  = str(rl_value).strip().lower()
    img = str(image_number_value).strip().lower()
    is_rt       = "rt" in rl
    is_temporal = "temporal" in img
    if is_rt and is_temporal:       return "rt_temporal"
    if (not is_rt) and is_temporal: return "lt_temporal"
    if is_rt:                       return "rt_otitis"
    return "lt_otitis"


def run(submission_path):
    ans_df  = pd.read_csv("val_set.csv").sort_values(by="No").reset_index(drop=True)
    pred_df = pd.read_csv(submission_path).sort_values(by="No").reset_index(drop=True)

    slice_cols = [str(i) for i in range(1, 133)]

    # 태스크별 y_true / y_pred 수집
    tasks = ["rt_temporal", "lt_temporal", "rt_otitis", "lt_otitis"]
    yt = {t: [] for t in tasks}
    yp = {t: [] for t in tasks}

    # 전체(기존 evaluation.py와 동일 집계)도 같이
    all_t, all_p = [], []

    for _, row in ans_df.iterrows():
        p_no = row["No"]
        task = task_of(row["R/L"], row["Image number"])

        pred_row = pred_df[
            (pred_df["No"] == p_no) &
            (pred_df["R/L"] == row["R/L"]) &
            (pred_df["Image number"] == row["Image number"])
        ]
        if pred_row.empty:
            continue
        pred_row = pred_row.iloc[0]

        for col in slice_cols:
            if pd.notna(row[col]):
                t_val = int(row[col])
                try:
                    v = pred_row[col]
                    p_val = -1 if (pd.isna(v) or v == "") else int(float(v))
                except Exception:
                    p_val = -1
                yt[task].append(t_val)
                yp[task].append(p_val)
                all_t.append(t_val)
                all_p.append(p_val)

    # ── 출력 ──
    print(f"\n채점 파일: {submission_path}")
    print("=" * 78)
    print(f"{'TASK':<14}{'N':>7}{'macroF1':>10}{'Acc':>9}"
          f"{'  | Normal: P/R/F1':<26}{'Otitis: P/R/F1':<22}")
    print("-" * 78)

    macro_f1_list = []
    for t in tasks:
        true = np.array(yt[t])
        pred = np.array(yp[t])
        if len(true) == 0:
            print(f"{t:<14}{'(데이터 없음)'}")
            continue

        n      = len(true)
        f1m    = f1_score(true, pred, average="macro", labels=[0, 1], zero_division=0)
        acc    = accuracy_score(true, pred)
        # 클래스별
        p0 = precision_score(true, pred, pos_label=0, zero_division=0)
        r0 = recall_score(true, pred, pos_label=0, zero_division=0)
        f0 = f1_score(true, pred, pos_label=0, zero_division=0)
        p1 = precision_score(true, pred, pos_label=1, zero_division=0)
        r1 = recall_score(true, pred, pos_label=1, zero_division=0)
        f1c = f1_score(true, pred, pos_label=1, zero_division=0)

        macro_f1_list.append(f1m)
        normal_str = f"{p0:.2f}/{r0:.2f}/{f0:.2f}"
        otitis_str = f"{p1:.2f}/{r1:.2f}/{f1c:.2f}"
        print(f"{t:<14}{n:>7}{f1m:>10.4f}{acc:>9.3f}"
              f"  | {normal_str:<24}{otitis_str:<22}")

    print("-" * 78)
    # 전체 (기존 evaluation.py 방식: 모든 슬라이스 합쳐 macro F1)
    at = np.array(all_t); ap = np.array(all_p)
    overall_f1  = f1_score(at, ap, average="macro", labels=[0, 1], zero_division=0)
    overall_acc = accuracy_score(at, ap)
    print(f"{'[전체 합산]':<14}{len(at):>7}{overall_f1:>10.4f}{overall_acc:>9.3f}"
          f"   (← 기존 evaluation.py와 동일 기준)")
    # 태스크 macro 평균 (참고용)
    if macro_f1_list:
        print(f"{'[태스크평균]':<14}{'':>7}{np.mean(macro_f1_list):>10.4f}"
              f"   (4개 태스크 macroF1의 단순 평균)")
    print("=" * 78)

    # 가장 약한 태스크 짚어주기
    if macro_f1_list:
        worst_idx = int(np.argmin(macro_f1_list))
        print(f"\n>>> 가장 약한 태스크: {tasks[worst_idx]} "
              f"(macroF1={macro_f1_list[worst_idx]:.4f})")
        print(">>> 이 태스크의 Normal/Otitis recall 중 낮은 쪽이 개선 1순위.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 evaluation_per_task.py submission_validation.csv")
    else:
        run(sys.argv[1])