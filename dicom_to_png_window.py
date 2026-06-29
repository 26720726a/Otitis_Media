"""
DICOM → window PNG 변환 스크립트 (TBCT)

목적:
    원본 DICOM의 HU값에서 측두골용 window를 직접 적용해 새 PNG를 생성한다.
    기존 PNG는 어떤 window로 구워졌는지 불명확하고 측두골에 안 맞을 수 있으므로,
    DICOM에서 bone window(default 700/4000)로 새로 "현상"한다.

    HU = raw_pixel × RescaleSlope + RescaleIntercept   (확인됨: slope=1, intercept=-1024)
    window range = [center - width/2, center + width/2] → 0~255 정규화

입출력 경로 (★ 기존 PNG는 절대 건드리지 않음):
    입력 : data/{train|val}/{pid}/DCM/{n:04d}.dcm
    출력 : data/{train|val}/{pid}/{OUT_SUBDIR}/{n:04d}.png   (기본 OUT_SUBDIR=PNG_bone)

사용법:
    python3 dicom_to_png_window.py                 # bone window(700/4000)로 전체 변환
    python3 dicom_to_png_window.py --soft          # soft-tissue window(35/250)로 변환
    python3 dicom_to_png_window.py --center 700 --width 4000 --out-subdir PNG_bone
    python3 dicom_to_png_window.py --overwrite      # 이미 있는 PNG도 다시 생성

변환 후:
    train_crop_3head.py / inference_crop_3head.py 에서 PNG 폴더명을
    "PNG" → "PNG_bone" 으로 바꾸면 됨. mean/std는 학습 시 자동 재계산되므로
    새 stats 파일로 떨어진다 (window가 바뀌면 픽셀 분포가 달라지므로 필수).
"""

import os
import glob
import argparse

import numpy as np
import cv2
import pydicom
from tqdm import tqdm


# 측두골 CT에 박혀있던 추천 window (DICOM 헤더 WindowCenter=[700,35], Width=[4000,250])
BONE_WINDOW = (700, 4000)   # 뼈/골미란 → temporal
SOFT_WINDOW = (35,  250)    # 연부조직/액체 → otitis


def find_data_root():
    for r in ("data", "./data", "../data"):
        if os.path.isdir(r):
            return r
    raise FileNotFoundError("data 디렉토리를 찾을 수 없습니다 (data / ./data / ../data).")


def dicom_to_windowed(path, center, width):
    """DICOM 1장 → window 적용된 uint8 grayscale (H, W)."""
    d   = pydicom.dcmread(path)
    arr = d.pixel_array.astype(np.float32)

    # 파일별 slope/intercept를 직접 읽어 적용 (환자/스캐너 차이 대비)
    slope = float(getattr(d, "RescaleSlope", 1) or 1)
    inter = float(getattr(d, "RescaleIntercept", 0) or 0)
    hu = arr * slope + inter

    lower = center - width / 2.0
    upper = center + width / 2.0
    img = (hu - lower) / (upper - lower)        # window 밖은 0 미만/1 초과
    img = np.clip(img, 0.0, 1.0)

    # MONOCHROME1이면 명암 반전 (CT는 보통 MONOCHROME2라 해당 없음, 안전장치)
    photo = str(getattr(d, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
    if photo == "MONOCHROME1":
        img = 1.0 - img

    return (img * 255.0).round().astype(np.uint8)


def convert_all(data_root, center, width, out_subdir, overwrite):
    # data/{split}/{pid}/DCM/*.dcm 전부 탐색
    dcm_dirs = sorted(glob.glob(os.path.join(data_root, "*", "*", "DCM")))
    if not dcm_dirs:
        # 혹시 split 없이 data/{pid}/DCM 구조인 경우도 대비
        dcm_dirs = sorted(glob.glob(os.path.join(data_root, "*", "DCM")))
    if not dcm_dirs:
        raise FileNotFoundError(f"DCM 폴더를 찾지 못했습니다. (예: {data_root}/train/10/DCM)")

    print(f"대상 DCM 폴더 {len(dcm_dirs)}개 | window=({center}/{width}) → '{out_subdir}/'")

    n_files = n_skip = n_err = 0
    first_stats_done = False

    for dcm_dir in tqdm(dcm_dirs, desc="환자 변환"):
        out_dir = os.path.join(os.path.dirname(dcm_dir), out_subdir)
        os.makedirs(out_dir, exist_ok=True)

        for dcm_path in sorted(glob.glob(os.path.join(dcm_dir, "*.dcm"))):
            base    = os.path.splitext(os.path.basename(dcm_path))[0]  # "0001"
            out_png = os.path.join(out_dir, f"{base}.png")

            if os.path.exists(out_png) and not overwrite:
                n_skip += 1
                continue

            try:
                img = dicom_to_windowed(dcm_path, center, width)
                cv2.imwrite(out_png, img)
                n_files += 1

                if not first_stats_done:
                    print(f"\n  [샘플 검증] {dcm_path}")
                    print(f"    출력 shape={img.shape}, dtype={img.dtype}, "
                          f"값범위={img.min()}~{img.max()}, 평균={img.mean():.1f}")
                    print(f"    저장: {out_png}\n")
                    first_stats_done = True
            except Exception as e:
                n_err += 1
                if n_err <= 5:
                    print(f"  [에러] {dcm_path}: {e}")

    print(f"\n변환 완료: 생성 {n_files} | 건너뜀(이미존재) {n_skip} | 에러 {n_err}")
    print(f"출력 폴더 예: {os.path.dirname(dcm_dirs[0])}/{out_subdir}/")
    print("\n다음 단계: train_crop_3head.py / inference_crop_3head.py 의 PNG 폴더명을")
    print(f"          'PNG' → '{out_subdir}' 로 바꾼 뒤 단일 fold 학습/추론.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DICOM → window PNG 변환 (기존 PNG 보존)")
    ap.add_argument("--soft", action="store_true",
                    help="soft-tissue window(35/250) 사용 (기본은 bone 700/4000)")
    ap.add_argument("--center", type=float, default=None, help="window center 직접 지정")
    ap.add_argument("--width",  type=float, default=None, help="window width 직접 지정")
    ap.add_argument("--out-subdir", type=str, default=None,
                    help="출력 폴더명 (기본: bone→PNG_bone, soft→PNG_soft)")
    ap.add_argument("--overwrite", action="store_true",
                    help="이미 존재하는 PNG도 다시 생성")
    args = ap.parse_args()

    if args.center is not None and args.width is not None:
        center, width = args.center, args.width
        out_subdir = args.out_subdir or f"PNG_c{int(center)}w{int(width)}"
    elif args.soft:
        center, width = SOFT_WINDOW
        out_subdir = args.out_subdir or "PNG_soft"
    else:
        center, width = BONE_WINDOW
        out_subdir = args.out_subdir or "PNG_bone"

    data_root = find_data_root()
    print(f"데이터 루트: {data_root}")
    convert_all(data_root, center, width, out_subdir, args.overwrite)