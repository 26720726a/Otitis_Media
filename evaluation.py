import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys


def run_grading(submission_path):
    try:
        ans_df  = pd.read_csv('val_set.csv')
        pred_df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"[Error] 파일 로드 에러: {e}")
        return

    ans_df  = ans_df.sort_values(by='No').reset_index(drop=True)
    pred_df = pred_df.sort_values(by='No').reset_index(drop=True)

    slice_cols = [str(i) for i in range(1, 133)]
    y_true, y_pred = [], []

    print(f"채점 시작: {submission_path}")

    for idx, row in ans_df.iterrows():
        p_no = row['No']

        # ── R/L, Image number까지 매칭하여 정확한 행 선택 ──
        pred_row = pred_df[
            (pred_df['No'] == p_no) &
            (pred_df['R/L'] == row['R/L']) &
            (pred_df['Image number'] == row['Image number'])
        ]

        if pred_row.empty:
            continue
        pred_row = pred_row.iloc[0]

        for col in slice_cols:
            if pd.notna(row[col]):
                y_true.append(int(row[col]))
                try:
                    val = pred_row[col]
                    if pd.isna(val) or val == "":
                        p_val = -1
                    else:
                        p_val = int(float(val))
                except:
                    p_val = -1
                y_pred.append(p_val)

    if len(y_true) == 0:
        print("[Error] 채점할 데이터가 없습니다.")
        return

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n" + "=" * 50)
    print("           TBCT ASSIGNMENT GRADING RESULT")
    print("=" * 50)
    print(f"  Target File    : {submission_path}")
    print(f"  Total Slices   : {len(y_true)}")
    print("-" * 50)
    print(f"  Accuracy Score : {acc * 100:.2f}%")
    print(f"  F1-Score (Avg) : {f1:.4f}")
    print("-" * 50)
    print("\n[ Detailed Report ]")

    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [
        ('Normal(0)' if l == 0 else 'Otitis(1)' if l == 1 else f'Invalid({l})')
        for l in unique_labels
    ]
    print(classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    ))
    print("=" * 50)
    print(f"  >>> FINAL SCORE: {acc * 100:.1f} / 100.0")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 evaluation.py submission_validation.csv")
    else:
        run_grading(sys.argv[1])