"""
pooled F1 직접 최적화 threshold 스윕 (model_convNeXt-Tiny_crop_2head 전용)

목적:
    실제 대회 점수 = evaluation.py 기준의 전체 합산(pooled) macroF1.
    4개 태스크 슬라이스를 하나의 confusion matrix로 합쳐 단 1번 계산하는 값.

    per-task 독립 최적(sweep_thresholds_2head.py)은 각 태스크 F1을 극대화하지만,
    pooled F1은 태스크별 F1의 평균이 아니므로 독립 최적 ≠ pooled 최적이다.
    이 스크립트는 4개 threshold를 동시에 움직여 pooled F1을 직접 최대화한다.

    확률 계산은 sweep_thresholds_2head.py를 import해 그대로 재사용한다.
    추론·스무딩·TASK_TO_CROP 매핑이 기존과 완전히 동일하고 추론은 딱 1번만 돈다.

    전수탐색: GRID 9개 × 4 태스크 = 9^4 = 6561 조합. 캐시된 확률만 재사용하므로 수 초면 끝남.

사용법:
    python3 sweep_thresholds_2head_pooled.py best_model_convNeXt-Tiny_crop_2head_fold0.pth

    ※ val_set.csv (정답 라벨)와 data/ 디렉토리가 있어야 함.
    ※ inference_crop_2head.py, sweep_thresholds_2head.py 가 같은 폴더에 있어야 함.
"""

import itertools
import os
import sys

import numpy as np
import torch

import sweep_thresholds_2head as base


# ==========================================================
# [태스크별 독립 최적 threshold 계산]
# ==========================================================
def find_indep_best(per_task):
    """각 태스크를 독립적으로 GRID 탐색 → 태스크별 best threshold dict 반환."""
    best_thr = {}
    for t in base.TASKS:
        true, prob = per_task[t]["true"], per_task[t]["prob"]
        if len(true) == 0:
            best_thr[t] = 0.50
            continue
        scores = [base.macro_f1(true, prob, g) for g in base.GRID]
        best_thr[t] = float(base.GRID[int(np.argmax(scores))])
    return best_thr


# ==========================================================
# [pooled F1 전수탐색]
#
# itertools.product(GRID, repeat=4) 로 9^4=6561개 조합을 전수 평가.
# 캐시된 확률(per_task)만 재사용하므로 추론 0회.
# ==========================================================
def find_pooled_best(per_task):
    best_score = -1.0
    best_combo = None

    for combo in itertools.product(base.GRID, repeat=4):
        thr_map = dict(zip(base.TASKS, combo))
        score   = base.overall_pooled_f1(per_task, thr_map)
        if score > best_score:
            best_score = score
            best_combo = thr_map

    return best_combo, best_score


# ==========================================================
# [메인]
# ==========================================================
def run(best_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    if not os.path.exists(base.VAL_CSV):
        print(f"[오류] {base.VAL_CSV} 없음")
        return

    infer = base.load_infer_module(base.INFER_FILE)

    # TASK_TO_CROP 점검 (2-head: head_idx >= 2 없어야 함)
    print("\n[점검] infer.TASK_TO_CROP 확인 (2-head: head_idx >= 2가 없어야 함)")
    for task_idx, task_name in enumerate(infer.OUTPUT_TASK_NAMES):
        side, head_idx = infer.TASK_TO_CROP[task_idx]
        marker = " ✓" if head_idx <= 1 else " ★★ 오류: head_idx >= 2"
        print(f"  task_idx={task_idx} ({task_name:12s}) → side={side!r:7s}, head_idx={head_idx}{marker}")

    models, stats = base.load_models_and_stats(infer, best_model_path, device)
    per_task      = base.collect_probs(infer, models, stats, device)

    # ── 1) baseline: 전부 0.50 ──
    base_map   = {t: 0.50 for t in base.TASKS}
    base_score = base.overall_pooled_f1(per_task, base_map)

    # ── 2) 태스크별 독립 최적 ──
    best_thr_indep = find_indep_best(per_task)
    indep_score    = base.overall_pooled_f1(per_task, best_thr_indep)

    # ── 3) pooled 전수탐색 ──
    print(f"\npooled F1 전수탐색 중 (GRID={len(base.GRID)}개 × 4태스크 "
          f"= {len(base.GRID)**4}조합) ...")
    best_combo, pooled_score = find_pooled_best(per_task)
    print("완료.")

    # 로직 검증: pooled 최적은 반드시 독립 최적 이상이어야 함
    # (best_thr_indep도 9^4 격자의 한 점이므로)
    assert pooled_score >= indep_score - 1e-9, (
        f"[버그] pooled 최적({pooled_score:.6f}) < 독립 최적({indep_score:.6f}). "
        "overall_pooled_f1 또는 find_pooled_best 로직을 확인하라."
    )

    # ── 결과 출력 ──
    SEP = "=" * 72

    print(f"\n{SEP}")
    print("3가지 방식 비교 (pooled macroF1 = 실제 대회 점수)")
    print(SEP)
    print(f"  {'방식':<28}  {'pooled F1':>10}  {'rt_temp':>8}  {'lt_temp':>8}  "
          f"{'rt_otitis':>9}  {'lt_otitis':>9}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")

    def _row(label, thr_map, score):
        return (f"  {label:<28}  {score:>10.4f}"
                f"  {thr_map['rt_temporal']:>8.2f}"
                f"  {thr_map['lt_temporal']:>8.2f}"
                f"  {thr_map['rt_otitis']:>9.2f}"
                f"  {thr_map['lt_otitis']:>9.2f}")

    print(_row("① baseline (전부 0.50)",    base_map,         base_score))
    print(_row("② 태스크별 독립 최적",       best_thr_indep,   indep_score))
    print(_row("③ pooled 직접 최적 ★",      best_combo,       pooled_score))
    print(SEP)

    gain_vs_base  = pooled_score - base_score
    gain_vs_indep = pooled_score - indep_score
    print(f"  pooled 최적 vs baseline  : {gain_vs_base:+.4f}")
    print(f"  pooled 최적 vs 독립 최적  : {gain_vs_indep:+.4f}")
    print(SEP)

    # ── 태스크별 F1 상세 (pooled 최적 threshold 적용 시) ──
    print(f"\n{SEP}")
    print("태스크별 F1 상세 (③ pooled 최적 threshold 적용)")
    print(SEP)
    print(f"  {'task':<14}  {'thr':>5}  {'F1(macro)':>10}  "
          f"{'precision':>10}  {'recall':>8}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")
    from sklearn.metrics import precision_score, recall_score
    for t in base.TASKS:
        true, prob = per_task[t]["true"], per_task[t]["prob"]
        thr        = best_combo[t]
        if len(true) == 0:
            print(f"  {t:<14}  {thr:>5.2f}  (데이터 없음)")
            continue
        pred = (prob >= thr).astype(int)
        f1   = base.macro_f1(true, prob, thr)
        prec = precision_score(true, pred, average="macro", labels=[0, 1], zero_division=0)
        rec  = recall_score(true,  pred, average="macro", labels=[0, 1], zero_division=0)
        print(f"  {t:<14}  {thr:>5.2f}  {f1:>10.4f}  {prec:>10.4f}  {rec:>8.4f}")
    print(SEP)

    # ── inference_crop_2head.py에 바로 붙여넣을 dict ──
    print(f"\n{SEP}")
    print("inference_crop_2head.py의 MANUAL_THRESHOLDS에 붙여넣을 값 (③ pooled 최적):")
    print(SEP)
    print("MANUAL_THRESHOLDS = {")
    for t in base.TASKS:
        print(f'    "{t}": {best_combo[t]:.2f},')
    print("}")
    print(SEP)

    # ★ 이 threshold는 단일 fold val에 과적합될 수 있음.
    #   최종 확정은 full 5-fold 앙상블에서 이 스윕을 다시 돌려 정할 것.
    print("\n[주의] 이 threshold는 단일 fold val에 과적합될 수 있음.")
    print("       최종 확정은 full 5-fold 앙상블에서 이 스윕을 다시 돌려 정할 것.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 sweep_thresholds_2head_pooled.py "
              "best_model_convNeXt-Tiny_crop_2head_fold0.pth")
        sys.exit(1)
    run(sys.argv[1])
