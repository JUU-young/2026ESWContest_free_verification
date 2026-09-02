from pathlib import Path

import cv2
import torch

BASE_DIR = Path(__file__).resolve().parent

YOLO_ROOT = BASE_DIR / "yolov5"
WEIGHTS = BASE_DIR / "best.pt"

CONF_THRESHOLD = 0.25
IMAGE_SIZE = 1280

TARGET_WIDTH = 1200
TARGET_HEIGHT = 620

def load_yolo_model():

    if not YOLO_ROOT.is_dir():
        raise FileNotFoundError(
            f"YOLOv5 폴더 없음: {YOLO_ROOT}"
        )

    if not (YOLO_ROOT / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"hubconf.py 없음: "
            f"{YOLO_ROOT / 'hubconf.py'}"
        )

    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            f"YOLO weight 없음: {WEIGHTS}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[YOLO] loading on {device}")

    model = torch.hub.load(
        str(YOLO_ROOT),
        "custom",
        path=str(WEIGHTS),
        source="local",
    )

    model.conf = CONF_THRESHOLD

    if device == "cuda":
        model = model.cuda()

    model.eval()

    print("[YOLO] ready")

    return model, device

def get_capture_images(capture_folder):

    capture_folder = Path(capture_folder)

    if not capture_folder.is_dir():
        return []

    images = []

    for path in capture_folder.iterdir():
        if not path.is_file():
            continue

        lower = path.name.lower()

        if "cam0" in lower or "cam1" in lower:
            images.append(path)

    return sorted(images)


def resize_crop(crop):

    crop_h, crop_w = crop.shape[:2]

    if crop_h <= 0 or crop_w <= 0:
        return None

    scale = min(
        TARGET_WIDTH / crop_w,
        TARGET_HEIGHT / crop_h,
    )

    new_w = max(
        1,
        int(round(crop_w * scale))
    )

    new_h = max(
        1,
        int(round(crop_h * scale))
    )

    interpolation = (
        cv2.INTER_LINEAR
        if scale >= 1.0
        else cv2.INTER_AREA
    )

    return cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=interpolation,
    )

def extract_best_signature(
    model,
    capture_folder,
    output_path,
    device,
):

    capture_folder = Path(capture_folder)
    output_path = Path(output_path)

    image_paths = get_capture_images(
        capture_folder
    )

    if not image_paths:
        print(
            f"[YOLO ERROR] 촬영 이미지 없음: "
            f"{capture_folder}"
        )
        return None, None

    best = None

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(
                f"[YOLO WARNING] 읽기 실패: "
                f"{image_path}"
            )
            continue

        image_h, image_w = image.shape[:2]

        print(
            f"[YOLO] inference: "
            f"{image_path.name} "
            f"({image_w}x{image_h})"
        )

        with torch.inference_mode():
            results = model(
                image,
                size=IMAGE_SIZE,
            )

        detections = (
            results.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )

        print(
            f"[YOLO] detections: "
            f"{len(detections)}"
        )

        for det in detections:

            x1, y1, x2, y2, conf, cls = det

            confidence = float(conf)

            if (
                best is not None
                and confidence <= best["confidence"]
            ):
                continue

            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(image_w, int(round(x2)))
            y2 = min(image_h, int(round(y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            best = {
                "confidence": confidence,
                "crop": crop.copy(),
                "source": image_path.name,
                "bbox": (x1, y1, x2, y2),
            }

    if best is None:
        print(
            "[YOLO ERROR] 검출된 서명이 없습니다."
        )
        return None, None

    resized = resize_crop(
        best["crop"]
    )

    if resized is None:
        print(
            "[YOLO ERROR] crop resize 실패"
        )
        return None, None

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

   
    success = cv2.imwrite(
        str(output_path),
        resized,
    )

    if not success:
        print(
            f"[YOLO ERROR] 저장 실패: "
            f"{output_path}"
        )
        return None, None

    print(
        "[YOLO BEST]",
        best["source"],
        f"conf={best['confidence']:.4f}",
        f"bbox={best['bbox']}",
        "->",
        output_path,
    )

    return (
        output_path,
        best["confidence"],
    )
