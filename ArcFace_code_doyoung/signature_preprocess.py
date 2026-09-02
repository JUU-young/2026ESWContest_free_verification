import cv2
import numpy as np
from torchvision import transforms


INPUT_WIDTH = 320
INPUT_HEIGHT = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


tensor_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


def resize_with_padding(
    img,
    target_w=INPUT_WIDTH,
    target_h=INPUT_HEIGHT
):

    if img is None:
        return None

    if img.size == 0:
        return None


    h, w = img.shape[:2]


    if h <= 0 or w <= 0:
        return None

    scale = min(
        target_w / w,
        target_h / h
    )


    new_w = max(
        1,
        int(
            round(
                w * scale
            )
        )
    )


    new_h = max(
        1,
        int(
            round(
                h * scale
            )
        )
    )


    if scale < 1.0:

        interpolation = (
            cv2.INTER_AREA
        )

    else:

        interpolation = (
            cv2.INTER_CUBIC
        )


    resized = cv2.resize(
        img,
        (
            new_w,
            new_h
        ),
        interpolation=interpolation
    )


    canvas = np.full(
        (
            target_h,
            target_w,
            3
        ),
        255,
        dtype=np.uint8
    )

    x = (
        target_w
        -
        new_w
    ) // 2


    y = (
        target_h
        -
        new_h
    ) // 2


    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized


    return canvas


def preprocess_cv_image(img):

    if img is None:
        return None


    if len(img.shape) == 2:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )


    elif (
        len(img.shape) == 3
        and
        img.shape[2] == 4
    ):

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGRA2BGR
        )

    img = resize_with_padding(
        img,
        INPUT_WIDTH,
        INPUT_HEIGHT
    )


    if img is None:
        return None

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )
    
    tensor = tensor_transform(
        img
    )


    return tensor

def preprocess_signature(
    image_path,
    add_batch_dimension=True
):

    img = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED
    )


    if img is None:

        print(
            f"[WARNING] 이미지 읽기 실패: "
            f"{image_path}"
        )

        return None


    tensor = preprocess_cv_image(
        img
    )


    if tensor is None:
        return None


    if add_batch_dimension:

        tensor = tensor.unsqueeze(
            0
        )


    return tensor