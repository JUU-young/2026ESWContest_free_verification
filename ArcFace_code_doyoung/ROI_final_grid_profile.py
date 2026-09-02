import os
import re
import cv2
import shutil
import tempfile
import json
import numpy as np

RUN_MODE = "batch"

CAM0_ROI_COLS = 2
CAM0_ROI_ROWS = 4

CAM1_ROI_COLS = 2
CAM1_ROI_ROWS = 3

CAM0_ROI_COUNT = CAM0_ROI_COLS * CAM0_ROI_ROWS
CAM1_ROI_COUNT = CAM1_ROI_COLS * CAM1_ROI_ROWS

AUTO_CONFIG_PROFILE = True
CONFIG_PROFILE_NAME = ""
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(BASE_DIR, "input")
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")

EXPECTED_WIDTH = 3264
EXPECTED_HEIGHT = 2448
PREVIEW_WIDTH = 820
PREVIEW_WINDOW_WIDTH = 590
PREVIEW_WINDOW_HEIGHT = 430
CONTROL_WINDOW_WIDTH = 590
CONTROL_WINDOW_HEIGHT = 850
CAM0_PREVIEW_X = 10
CAM0_PREVIEW_Y = 20
CAM1_PREVIEW_X = 620
CAM1_PREVIEW_Y = 20
CAM0_CONTROL_X = 10
CAM0_CONTROL_Y = 480
CAM1_CONTROL_X = 620
CAM1_CONTROL_Y = 480
INNER_MARGIN = 45
OUTPUT_SIZE = None

CAM0_K1 = -0.1778
CAM0_K2 = 0.0208
CAM0_K3 = 0.0069
CAM0_CENTER_X = 1733
CAM0_CENTER_Y = 1102
CAM0_FOCAL = 2380
CAM0_ROTATION = 0.0

CAM1_K1 = -0.1409
CAM1_K2 = 0.0577
CAM1_K3 = -0.0254
CAM1_CENTER_X = 1718
CAM1_CENTER_Y = 1277
CAM1_FOCAL = 2469
CAM1_ROTATION = 0.0

CAM0_X_LEFT = 610
CAM0_X_RIGHT = 2690
CAM0_Y_TOP = 774
CAM0_Y_BOTTOM = 2362

CAM1_X_LEFT = 324
CAM1_X_RIGHT = 2720
CAM1_Y_TOP = 667
CAM1_Y_BOTTOM = 2046

CAM0_X_BOUNDARIES = []
CAM0_Y_BOUNDARIES = []
CAM1_X_BOUNDARIES = []
CAM1_Y_BOUNDARIES = []

K_MAX_ABS = 1.0
K_SLIDER_CENTER = 10000
K_SLIDER_MAX = 20000
FOCAL_MIN = 500
FOCAL_MAX = 6000
ROTATION_MIN = -30
ROTATION_MAX = 30

def get_profile_path():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    if AUTO_CONFIG_PROFILE or not CONFIG_PROFILE_NAME.strip():
        name = f"cam0_{CAM0_ROI_COLS}x{CAM0_ROI_ROWS}_cam1_{CAM1_ROI_COLS}x{CAM1_ROI_ROWS}"
    else:
        name = CONFIG_PROFILE_NAME.strip()
    if not name.lower().endswith(".json"):
        name += ".json"
    return os.path.join(PROFILE_DIR, name)

def _set_global_config(data):
    global CAM0_K1, CAM0_K2, CAM0_K3, CAM0_CENTER_X, CAM0_CENTER_Y, CAM0_FOCAL, CAM0_ROTATION
    global CAM1_K1, CAM1_K2, CAM1_K3, CAM1_CENTER_X, CAM1_CENTER_Y, CAM1_FOCAL, CAM1_ROTATION
    global CAM0_X_BOUNDARIES, CAM0_Y_BOUNDARIES, CAM1_X_BOUNDARIES, CAM1_Y_BOUNDARIES
    global CAM0_X_LEFT, CAM0_X_RIGHT, CAM0_Y_TOP, CAM0_Y_BOTTOM
    global CAM1_X_LEFT, CAM1_X_RIGHT, CAM1_Y_TOP, CAM1_Y_BOTTOM

    float_keys = [
        "CAM0_K1","CAM0_K2","CAM0_K3","CAM0_ROTATION",
        "CAM1_K1","CAM1_K2","CAM1_K3","CAM1_ROTATION"
    ]
    int_keys = [
        "CAM0_CENTER_X","CAM0_CENTER_Y","CAM0_FOCAL",
        "CAM1_CENTER_X","CAM1_CENTER_Y","CAM1_FOCAL"
    ]
    for key in float_keys:
        if key in data:
            globals()[key] = float(data[key])
    for key in int_keys:
        if key in data:
            globals()[key] = int(data[key])

    for prefix in ("CAM0", "CAM1"):
        xkey = f"{prefix}_X_BOUNDARIES"
        ykey = f"{prefix}_Y_BOUNDARIES"
        if xkey in data:
            globals()[xkey] = [int(v) for v in data[xkey]]
        if ykey in data:
            globals()[ykey] = [int(v) for v in data[ykey]]

    if len(CAM0_X_BOUNDARIES) >= 2:
        CAM0_X_LEFT, CAM0_X_RIGHT = CAM0_X_BOUNDARIES[0], CAM0_X_BOUNDARIES[-1]
    if len(CAM0_Y_BOUNDARIES) >= 2:
        CAM0_Y_TOP, CAM0_Y_BOTTOM = CAM0_Y_BOUNDARIES[0], CAM0_Y_BOUNDARIES[-1]
    if len(CAM1_X_BOUNDARIES) >= 2:
        CAM1_X_LEFT, CAM1_X_RIGHT = CAM1_X_BOUNDARIES[0], CAM1_X_BOUNDARIES[-1]
    if len(CAM1_Y_BOUNDARIES) >= 2:
        CAM1_Y_TOP, CAM1_Y_BOTTOM = CAM1_Y_BOUNDARIES[0], CAM1_Y_BOUNDARIES[-1]

def generate_default_boundaries(count, start, end):
    if count < 1:
        return [int(start), int(end)]
    return [int(round(v)) for v in np.linspace(start, end, count + 1)]

def load_active_profile():
    global CAM0_X_BOUNDARIES, CAM0_Y_BOUNDARIES, CAM1_X_BOUNDARIES, CAM1_Y_BOUNDARIES
    path = get_profile_path()
    if not os.path.isfile(path):
        CAM0_X_BOUNDARIES = generate_default_boundaries(CAM0_ROI_COLS, CAM0_X_LEFT, CAM0_X_RIGHT)
        CAM0_Y_BOUNDARIES = generate_default_boundaries(CAM0_ROI_ROWS, CAM0_Y_TOP, CAM0_Y_BOTTOM)
        CAM1_X_BOUNDARIES = generate_default_boundaries(CAM1_ROI_COLS, CAM1_X_LEFT, CAM1_X_RIGHT)
        CAM1_Y_BOUNDARIES = generate_default_boundaries(CAM1_ROI_ROWS, CAM1_Y_TOP, CAM1_Y_BOTTOM)
        print(f"[PROFILE] 기존 프로파일 없음 -> 기본값 사용: {path}")
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (int(data.get("CAM0_ROI_COLS", -1)) != CAM0_ROI_COLS or
            int(data.get("CAM0_ROI_ROWS", -1)) != CAM0_ROI_ROWS or
            int(data.get("CAM1_ROI_COLS", -1)) != CAM1_ROI_COLS or
            int(data.get("CAM1_ROI_ROWS", -1)) != CAM1_ROI_ROWS):
            print("[PROFILE] ROI GRID가 달라 프로파일을 사용하지 않습니다.")
            CAM0_X_BOUNDARIES = generate_default_boundaries(CAM0_ROI_COLS, CAM0_X_LEFT, CAM0_X_RIGHT)
            CAM0_Y_BOUNDARIES = generate_default_boundaries(CAM0_ROI_ROWS, CAM0_Y_TOP, CAM0_Y_BOTTOM)
            CAM1_X_BOUNDARIES = generate_default_boundaries(CAM1_ROI_COLS, CAM1_X_LEFT, CAM1_X_RIGHT)
            CAM1_Y_BOUNDARIES = generate_default_boundaries(CAM1_ROI_ROWS, CAM1_Y_TOP, CAM1_Y_BOTTOM)
            return False
        _set_global_config(data)
        print(f"[PROFILE] 불러옴: {path}")
        return True
    except Exception as e:
        print(f"[PROFILE] 불러오기 실패: {e}")
        return False

def get_profile_data(distortion_config, roi_config):
    return {**distortion_config, **roi_config,
            "CAM0_ROI_COLS": CAM0_ROI_COLS,
            "CAM0_ROI_ROWS": CAM0_ROI_ROWS,
            "CAM1_ROI_COLS": CAM1_ROI_COLS,
            "CAM1_ROI_ROWS": CAM1_ROI_ROWS,
            "CAM0_ROI_COUNT": CAM0_ROI_COUNT,
            "CAM1_ROI_COUNT": CAM1_ROI_COUNT,
            "version": 3}

def save_profile(distortion_config, roi_config):
    if not validate_roi_config(roi_config):
        print("[ERROR] ROI 좌표 순서가 잘못되어 저장하지 않습니다.")
        return False
    path = get_profile_path()
    os.makedirs(PROFILE_DIR, exist_ok=True)
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(get_profile_data(distortion_config, roi_config), f, ensure_ascii=False, indent=4)
        os.replace(temp_path, path)
        print(f"[PROFILE] 저장 완료: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] 프로파일 저장 실패: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return False

def natural_sort_key(text):

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]

def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )

def imread_korean(path):

    try:

        data = np.fromfile(
            path,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception as e:

        print("[ERROR] 이미지 읽기 실패")
        print(path)
        print(e)

        return None


def imwrite_korean(
    path,
    image,
    ext=".png"
):

    if image is None or image.size == 0:
        return False

    try:

        success, encoded = cv2.imencode(
            ext,
            image
        )

        if not success:
            return False

        encoded.tofile(path)

        return True

    except Exception as e:

        print("[ERROR] 이미지 저장 실패")
        print(path)
        print(e)

        return False


def get_image_files(folder):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    if not os.path.isdir(folder):
        return []

    files = []

    for name in os.listdir(folder):

        if name.lower().endswith(
            extensions
        ):

            files.append(name)

    return sorted(
        files,
        key=natural_sort_key
    )


def discover_input_sets():

    if not os.path.isdir(INPUT_ROOT):

        print()
        print("[ERROR] input 폴더가 없습니다.")
        print(INPUT_ROOT)

        return []

    sets = []

    root_images = get_image_files(
        INPUT_ROOT
    )

    if root_images:

        sets.append(
            (
                "input_root",
                INPUT_ROOT
            )
        )
        
        
    child_names = []

    for name in os.listdir(INPUT_ROOT):

        path = os.path.join(
            INPUT_ROOT,
            name
        )

        if not os.path.isdir(path):
            continue

        if name.startswith("."):
            continue

        if name.startswith("_"):
            continue

        child_names.append(name)

    child_names = sorted(
        child_names,
        key=natural_sort_key
    )

    for name in child_names:

        sets.append(
            (
                name,
                os.path.join(
                    INPUT_ROOT,
                    name
                )
            )
        )

    return sets


def load_camera_images(
    folder_path,
    verbose=True
):

    files = get_image_files(
        folder_path
    )

    cam0_files = []
    cam1_files = []

    for file_name in files:

        lower = file_name.lower()

        has_cam0 = "cam0" in lower
        has_cam1 = "cam1" in lower

        if has_cam0 and has_cam1:

            print()
            print(
                "[ERROR] 파일명에 cam0과 cam1이 "
                "동시에 포함되어 있습니다."
            )

            print(file_name)

            return None, None

        if has_cam0:

            cam0_files.append(
                file_name
            )

        elif has_cam1:

            cam1_files.append(
                file_name
            )

    if len(cam0_files) != 1:

        if verbose:

            print(
                f"[ERROR] CAM0 이미지 개수: "
                f"{len(cam0_files)}"
            )

        return None, None

    if len(cam1_files) != 1:

        if verbose:

            print(
                f"[ERROR] CAM1 이미지 개수: "
                f"{len(cam1_files)}"
            )

        return None, None

    if verbose:

        print(
            "  CAM0:",
            cam0_files[0]
        )

        print(
            "  CAM1:",
            cam1_files[0]
        )

    cam0_path = os.path.join(
        folder_path,
        cam0_files[0]
    )

    cam1_path = os.path.join(
        folder_path,
        cam1_files[0]
    )

    return (
        imread_korean(
            cam0_path
        ),

        imread_korean(
            cam1_path
        )
    )


def check_resolution(
    image,
    camera_name,
    verbose=True
):

    if image is None:
        return False

    height, width = image.shape[:2]

    if verbose:

        print(
            f"  {camera_name}: "
            f"{width} x {height}"
        )

    if (
        width != EXPECTED_WIDTH
        or
        height != EXPECTED_HEIGHT
    ):

        print()
        print(
            f"[ERROR] {camera_name} 해상도 오류"
        )

        print(
            f"현재: {width} x {height}"
        )

        print(
            f"예상: "
            f"{EXPECTED_WIDTH} x {EXPECTED_HEIGHT}"
        )

        return False

    return True

def find_first_valid_input():

    sets = discover_input_sets()

    for name, folder_path in sets:

        print()
        print(
            "[Calibration 입력]",
            name
        )

        cam0, cam1 = load_camera_images(
            folder_path
        )

        if (
            cam0 is None
            or
            cam1 is None
        ):
            continue

        if not check_resolution(
            cam0,
            "CAM0"
        ):
            continue

        if not check_resolution(
            cam1,
            "CAM1"
        ):
            continue

        return (
            name,
            cam0,
            cam1
        )

    return (
        None,
        None,
        None
    )


def get_current_distortion_config():
    return {
        "CAM0_K1": float(CAM0_K1), "CAM0_K2": float(CAM0_K2), "CAM0_K3": float(CAM0_K3),
        "CAM0_CENTER_X": int(CAM0_CENTER_X), "CAM0_CENTER_Y": int(CAM0_CENTER_Y), "CAM0_FOCAL": int(CAM0_FOCAL), "CAM0_ROTATION": float(CAM0_ROTATION),
        "CAM1_K1": float(CAM1_K1), "CAM1_K2": float(CAM1_K2), "CAM1_K3": float(CAM1_K3),
        "CAM1_CENTER_X": int(CAM1_CENTER_X), "CAM1_CENTER_Y": int(CAM1_CENTER_Y), "CAM1_FOCAL": int(CAM1_FOCAL), "CAM1_ROTATION": float(CAM1_ROTATION),
    }


def get_current_roi_config():
    return {
        "CAM0_X_BOUNDARIES": [int(v) for v in CAM0_X_BOUNDARIES],
        "CAM0_Y_BOUNDARIES": [int(v) for v in CAM0_Y_BOUNDARIES],
        "CAM1_X_BOUNDARIES": [int(v) for v in CAM1_X_BOUNDARIES],
        "CAM1_Y_BOUNDARIES": [int(v) for v in CAM1_Y_BOUNDARIES],
    }


def validate_roi_config(config):
    expected0x = CAM0_ROI_COLS + 1
    expected0y = CAM0_ROI_ROWS + 1
    expected1x = CAM1_ROI_COLS + 1
    expected1y = CAM1_ROI_ROWS + 1
    b0x = config["CAM0_X_BOUNDARIES"]; b0y = config["CAM0_Y_BOUNDARIES"]
    b1x = config["CAM1_X_BOUNDARIES"]; b1y = config["CAM1_Y_BOUNDARIES"]
    def valid(arr, expected, maximum):
        return len(arr) == expected and all(0 <= arr[i] < arr[i+1] <= maximum for i in range(len(arr)-1))
    return (valid(b0x, expected0x, EXPECTED_WIDTH) and
            valid(b0y, expected0y, EXPECTED_HEIGHT) and
            valid(b1x, expected1x, EXPECTED_WIDTH) and
            valid(b1y, expected1y, EXPECTED_HEIGHT))


def replace_block_in_text(
    source,
    start_marker,
    end_marker,
    new_block
):

    start_index = source.find(
        start_marker
    )

    end_index = source.find(
        end_marker
    )

    if start_index == -1:

        raise RuntimeError(
            f"START marker 없음: {start_marker}"
        )

    if end_index == -1:

        raise RuntimeError(
            f"END marker 없음: {end_marker}"
        )

    if end_index <= start_index:

        raise RuntimeError(
            "Marker 순서 오류"
        )

    end_index += len(
        end_marker
    )

    return (
        source[:start_index]
        +
        new_block
        +
        source[end_index:]
    )


def save_all_config_to_code(distortion_config, roi_config):
    return save_profile(distortion_config, roi_config)


def k_to_slider(value):

    value = clamp(
        float(value),
        -K_MAX_ABS,
        K_MAX_ABS
    )

    return int(
        round(
            (
                value
                +
                K_MAX_ABS
            )
            *
            10000
        )
    )


def slider_to_k(value):

    return (
        float(value)
        -
        K_SLIDER_CENTER
    ) / 10000.0



def resize_preview(
    image
):

    height, width = image.shape[:2]

    scale = (
        PREVIEW_WIDTH
        /
        float(width)
    )

    preview_height = int(
        round(
            height
            *
            scale
        )
    )

    return cv2.resize(
        image,
        (
            PREVIEW_WIDTH,
            preview_height
        ),
        interpolation=cv2.INTER_AREA
    )


def create_camera_matrix(
    focal,
    center_x,
    center_y,
    image_width,
    image_height
):

    scale_x = (
        image_width
        /
        float(EXPECTED_WIDTH)
    )

    scale_y = (
        image_height
        /
        float(EXPECTED_HEIGHT)
    )

    fx = (
        focal
        *
        scale_x
    )

    fy = (
        focal
        *
        scale_y
    )

    cx = (
        center_x
        *
        scale_x
    )

    cy = (
        center_y
        *
        scale_y
    )

    return np.array(
        [
            [
                fx,
                0.0,
                cx
            ],

            [
                0.0,
                fy,
                cy
            ],

            [
                0.0,
                0.0,
                1.0
            ]
        ],
        dtype=np.float64
    )


def get_camera_distortion_values(
    config,
    camera
):

    prefix = (
        "CAM0"
        if camera == 0
        else "CAM1"
    )

    return {

        "k1":
            float(
                config[
                    f"{prefix}_K1"
                ]
            ),

        "k2":
            float(
                config[
                    f"{prefix}_K2"
                ]
            ),

        "k3":
            float(
                config[
                    f"{prefix}_K3"
                ]
            ),

        "center_x":
            int(
                config[
                    f"{prefix}_CENTER_X"
                ]
            ),

        "center_y":
            int(
                config[
                    f"{prefix}_CENTER_Y"
                ]
            ),

        "focal":
            int(
                config[
                    f"{prefix}_FOCAL"
                ]
            ),

        "rotation":
            float(
                config[
                    f"{prefix}_ROTATION"
                ]
            )
    }

def create_undistort_map(
    width,
    height,
    values
):

    camera_matrix = create_camera_matrix(
        values["focal"],
        values["center_x"],
        values["center_y"],
        width,
        height
    )

    dist_coeffs = np.array(
        [
            values["k1"],
            values["k2"],
            0.0,
            0.0,
            values["k3"]
        ],
        dtype=np.float64
    )

    map1, map2 = (
        cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
            (
                width,
                height
            ),
            cv2.CV_16SC2
        )
    )

    return (
        map1,
        map2
    )

def apply_undistort(
    image,
    map1,
    map2
):

    return cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

def rotate_with_config(image, rotation):
    angle = float(rotation)
    if abs(angle) < 1e-9:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def undistort_with_config(
    image,
    distortion_config,
    camera
):

    height, width = image.shape[:2]

    values = (
        get_camera_distortion_values(
            distortion_config,
            camera
        )
    )

    map1, map2 = (
        create_undistort_map(
            width,
            height,
            values
        )
    )

    return apply_undistort(
        image,
        map1,
        map2
    )

def make_grid_rois(x_boundaries, y_boundaries):
    rois = []
    for row in range(len(y_boundaries) - 1):
        y1, y2 = y_boundaries[row], y_boundaries[row + 1]
        for col in range(len(x_boundaries) - 1):
            x1, x2 = x_boundaries[col], x_boundaries[col + 1]
            rois.append((x1, y1, x2, y2))
    return rois

def make_cam0_rois(config):
    return make_grid_rois(config["CAM0_X_BOUNDARIES"], config["CAM0_Y_BOUNDARIES"])

def make_cam1_rois(config):
    return make_grid_rois(config["CAM1_X_BOUNDARIES"], config["CAM1_Y_BOUNDARIES"])


def draw_roi_preview(
    image,
    roi_list,
    start_number
):

    result = image.copy()

    height, width = result.shape[:2]

    scale_x = (
        width
        /
        float(EXPECTED_WIDTH)
    )

    scale_y = (
        height
        /
        float(EXPECTED_HEIGHT)
    )

    for index, roi in enumerate(
        roi_list
    ):

        x1, y1, x2, y2 = roi

        px1 = int(
            round(
                x1 * scale_x
            )
        )

        py1 = int(
            round(
                y1 * scale_y
            )
        )

        px2 = int(
            round(
                x2 * scale_x
            )
        )

        py2 = int(
            round(
                y2 * scale_y
            )
        )

        cv2.rectangle(
            result,
            (
                px1,
                py1
            ),
            (
                px2,
                py2
            ),
            (
                0,
                0,
                255
            ),
            3
        )

        number = (
            start_number
            +
            index
        )

        cv2.putText(
            result,
            str(number),
            (
                px1 + 10,
                py1 + 32
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (
                0,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )

    return result

def draw_lens_info(
    image,
    values
):

    result = image.copy()

    texts = [

        (
            f"K1 {values['k1']:+.4f}   "
            f"K2 {values['k2']:+.4f}   "
            f"K3 {values['k3']:+.4f}"
        ),

        (
            f"CENTER "
            f"({values['center_x']}, "
            f"{values['center_y']})   "
            f"FOCAL {values['focal']}   "
            f"ROT {values.get('rotation', 0.0):+.1f}"
        )
    ]

    y = 25

    for text in texts:

        cv2.putText(
            result,
            text,
            (
                10,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                0,
                0
            ),
            4,
            cv2.LINE_AA
        )

        cv2.putText(
            result,
            text,
            (
                10,
                y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )

        y += 25

    return result

def nothing(value):
    pass

def create_camera_controls(control_window, distortion_config, roi_config, camera):
    prefix = "CAM0" if camera == 0 else "CAM1"
    for name in ("K1", "K2", "K3"):
        cv2.createTrackbar(name, control_window, k_to_slider(distortion_config[f"{prefix}_{name}"]), K_SLIDER_MAX, nothing)
    cv2.createTrackbar("CENTER X", control_window, clamp(distortion_config[f"{prefix}_CENTER_X"], 0, EXPECTED_WIDTH-1), EXPECTED_WIDTH-1, nothing)
    cv2.createTrackbar("CENTER Y", control_window, clamp(distortion_config[f"{prefix}_CENTER_Y"], 0, EXPECTED_HEIGHT-1), EXPECTED_HEIGHT-1, nothing)
    cv2.createTrackbar("FOCAL", control_window, clamp(distortion_config[f"{prefix}_FOCAL"], FOCAL_MIN, FOCAL_MAX)-FOCAL_MIN, FOCAL_MAX-FOCAL_MIN, nothing)
    cv2.createTrackbar("ROTATION", control_window, int(round(clamp(distortion_config[f"{prefix}_ROTATION"], ROTATION_MIN, ROTATION_MAX))) - ROTATION_MIN, ROTATION_MAX-ROTATION_MIN, nothing)
    xb = roi_config[f"{prefix}_X_BOUNDARIES"]
    yb = roi_config[f"{prefix}_Y_BOUNDARIES"]
    for i, value in enumerate(xb):
        cv2.createTrackbar(f"ROI X {i:02d}", control_window, clamp(value, 0, EXPECTED_WIDTH-1), EXPECTED_WIDTH-1, nothing)
    for i, value in enumerate(yb):
        cv2.createTrackbar(f"ROI Y {i:02d}", control_window, clamp(value, 0, EXPECTED_HEIGHT-1), EXPECTED_HEIGHT-1, nothing)

def read_camera_controls(control_window, camera, distortion_config, roi_config):
    prefix = "CAM0" if camera == 0 else "CAM1"
    for name in ("K1", "K2", "K3"):
        distortion_config[f"{prefix}_{name}"] = slider_to_k(cv2.getTrackbarPos(name, control_window))
    distortion_config[f"{prefix}_CENTER_X"] = cv2.getTrackbarPos("CENTER X", control_window)
    distortion_config[f"{prefix}_CENTER_Y"] = cv2.getTrackbarPos("CENTER Y", control_window)
    distortion_config[f"{prefix}_FOCAL"] = cv2.getTrackbarPos("FOCAL", control_window) + FOCAL_MIN
    distortion_config[f"{prefix}_ROTATION"] = cv2.getTrackbarPos("ROTATION", control_window) + ROTATION_MIN
    x_count = CAM0_ROI_COLS + 1 if camera == 0 else CAM1_ROI_COLS + 1
    y_count = CAM0_ROI_ROWS + 1 if camera == 0 else CAM1_ROI_ROWS + 1
    roi_config[f"{prefix}_X_BOUNDARIES"] = [cv2.getTrackbarPos(f"ROI X {i:02d}", control_window) for i in range(x_count)]
    roi_config[f"{prefix}_Y_BOUNDARIES"] = [cv2.getTrackbarPos(f"ROI Y {i:02d}", control_window) for i in range(y_count)]

def make_control_panel_image(
    camera_name
):

    panel = np.zeros(
        (
            90,
            CONTROL_WINDOW_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    cv2.putText(
        panel,
        f"{camera_name} Lens + ROI",
        (
            15,
            32
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        "S = SAVE ALL    Q / ESC = EXIT",
        (
            15,
            65
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )

    return panel

def run_calibration():

    source_name, cam0, cam1 = (
        find_first_valid_input()
    )

    if (
        cam0 is None
        or
        cam1 is None
    ):

        print()
        print(
            "[ERROR] Calibration에 사용할 "
            "CAM0/CAM1 이미지를 찾지 못했습니다."
        )

        return

    base0 = resize_preview(
        cam0
    )

    base1 = resize_preview(
        cam1
    )


    distortion_config = (
        get_current_distortion_config()
    )

    roi_config = (
        get_current_roi_config()
    )


    preview0_window = (
        f"CAM0 Preview - {source_name}"
    )

    preview1_window = (
        f"CAM1 Preview - {source_name}"
    )

    control0_window = (
        "CAM0 Controls"
    )

    control1_window = (
        "CAM1 Controls"
    )


    cv2.namedWindow(
        preview0_window,
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        preview1_window,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        preview0_window,
        PREVIEW_WINDOW_WIDTH,
        PREVIEW_WINDOW_HEIGHT
    )

    cv2.resizeWindow(
        preview1_window,
        PREVIEW_WINDOW_WIDTH,
        PREVIEW_WINDOW_HEIGHT
    )

    cv2.moveWindow(
        preview0_window,
        CAM0_PREVIEW_X,
        CAM0_PREVIEW_Y
    )

    cv2.moveWindow(
        preview1_window,
        CAM1_PREVIEW_X,
        CAM1_PREVIEW_Y
    )


    cv2.namedWindow(
        control0_window,
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        control1_window,
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        control0_window,
        make_control_panel_image(
            "CAM0"
        )
    )

    cv2.imshow(
        control1_window,
        make_control_panel_image(
            "CAM1"
        )
    )

    create_camera_controls(
        control0_window,
        distortion_config,
        roi_config,
        0
    )

    create_camera_controls(
        control1_window,
        distortion_config,
        roi_config,
        1
    )

    cv2.resizeWindow(
        control0_window,
        CONTROL_WINDOW_WIDTH,
        CONTROL_WINDOW_HEIGHT
    )

    cv2.resizeWindow(
        control1_window,
        CONTROL_WINDOW_WIDTH,
        CONTROL_WINDOW_HEIGHT
    )

    cv2.moveWindow(
        control0_window,
        CAM0_CONTROL_X,
        CAM0_CONTROL_Y
    )

    cv2.moveWindow(
        control1_window,
        CAM1_CONTROL_X,
        CAM1_CONTROL_Y
    )

    print()
    print("============================================")
    print("Lens + ROI Calibration")
    print("============================================")

    print()
    print("추천 순서:")
    print()
    print("1. K1")
    print("민성_test. K2")
    print("3. 필요하면 K3")
    print("4. 필요하면 CENTER X/Y")
    print("5. 필요하면 FOCAL")
    print("6. ROI X/Y 경계")

    print()
    print("S   : Lens + Rotation + ROI를 프로파일(JSON)에 저장")
    print("Q   : 종료")
    print("ESC : 종료")

    print()
    print("현재 시작 Lens 값:")

    print(
        "CAM0:",
        CAM0_K1,
        CAM0_K2,
        CAM0_K3
    )

    print(
        "CAM1:",
        CAM1_K1,
        CAM1_K2,
        CAM1_K3
    )

    print("============================================")

    last_lens0 = None
    last_lens1 = None

    last_roi0 = None
    last_roi1 = None

    corrected0 = base0.copy()
    corrected1 = base1.copy()

    display0 = base0.copy()
    display1 = base1.copy()

    while True:

        read_camera_controls(
            control0_window,
            0,
            distortion_config,
            roi_config
        )

        read_camera_controls(
            control1_window,
            1,
            distortion_config,
            roi_config
        )

        lens0 = get_camera_distortion_values(distortion_config, 0)
        lens0["rotation"] = float(distortion_config["CAM0_ROTATION"])

        lens1 = get_camera_distortion_values(distortion_config, 1)
        lens1["rotation"] = float(distortion_config["CAM1_ROTATION"])
        
        roi0 = dict(roi_config)
        roi1 = dict(roi_config)

        lens0_changed = (
            lens0
            !=
            last_lens0
        )

        if lens0_changed:
            corrected0 = undistort_with_config(base0, distortion_config, 0)
            corrected0 = rotate_with_config(corrected0, distortion_config["CAM0_ROTATION"])
            last_lens0 = lens0

        lens1_changed = (
            lens1
            !=
            last_lens1
        )

        if lens1_changed:
            corrected1 = undistort_with_config(base1, distortion_config, 1)
            corrected1 = rotate_with_config(corrected1, distortion_config["CAM1_ROTATION"])
            last_lens1 = lens1

        if (
            lens0_changed
            or
            roi0 != last_roi0
        ):

            display0 = draw_roi_preview(
                corrected0,
                make_cam0_rois(
                    roi_config
                ),
                1
            )

            display0 = draw_lens_info(
                display0,
                lens0
            )

            last_roi0 = (
                roi0.copy()
            )

        if (
            lens1_changed
            or
            roi1 != last_roi1
        ):

            display1 = draw_roi_preview(
                corrected1,
                make_cam1_rois(roi_config),
                CAM0_ROI_COUNT + 1
            )

            display1 = draw_lens_info(
                display1,
                lens1
            )

            last_roi1 = (
                roi1.copy()
            )

        if not validate_roi_config(
            roi_config
        ):

            warning0 = (
                display0.copy()
            )

            warning1 = (
                display1.copy()
            )

            cv2.putText(
                warning0,
                "INVALID ROI ORDER",
                (
                    15,
                    90
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    0,
                    255
                ),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                warning1,
                "INVALID ROI ORDER",
                (
                    15,
                    90
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (
                    0,
                    0,
                    255
                ),
                3,
                cv2.LINE_AA
            )

            cv2.imshow(
                preview0_window,
                warning0
            )

            cv2.imshow(
                preview1_window,
                warning1
            )

        else:

            cv2.imshow(
                preview0_window,
                display0
            )

            cv2.imshow(
                preview1_window,
                display1
            )


        key = (
            cv2.waitKey(15)
            &
            0xFF
        )
        
        
        if (
            key == ord("s")
            or
            key == ord("S")
        ):

            save_all_config_to_code(
                distortion_config,
                roi_config
            )

        elif (
            key == ord("q")
            or
            key == ord("Q")
            or
            key == 27
        ):

            break


    cv2.destroyAllWindows()

def crop_single_roi(
    image,
    roi
):

    x1, y1, x2, y2 = roi

    height, width = (
        image.shape[:2]
    )

    x1 += INNER_MARGIN
    y1 += INNER_MARGIN

    x2 -= INNER_MARGIN
    y2 -= INNER_MARGIN

    if (
        x1 < 0
        or
        y1 < 0
        or
        x2 > width
        or
        y2 > height
        or
        x1 >= x2
        or
        y1 >= y2
    ):

        return None

    cropped = image[
        y1:y2,
        x1:x2
    ].copy()

    if cropped.size == 0:
        return None

    if OUTPUT_SIZE is not None:

        cropped = cv2.resize(
            cropped,
            OUTPUT_SIZE,
            interpolation=cv2.INTER_AREA
        )

    return cropped

def extract_rois(cam0, cam1, roi_config):
    results = []
    for roi in make_cam0_rois(roi_config):
        cropped = crop_single_roi(cam0, roi)
        if cropped is None: return None
        results.append(cropped)
    for roi in make_cam1_rois(roi_config):
        cropped = crop_single_roi(cam1, roi)
        if cropped is None: return None
        results.append(cropped)
    expected = CAM0_ROI_COUNT + CAM1_ROI_COUNT
    return results if len(results) == expected else None

def get_next_output_number():

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    numbers = []

    for name in os.listdir(
        OUTPUT_ROOT
    ):

        path = os.path.join(
            OUTPUT_ROOT,
            name
        )

        if (
            os.path.isdir(path)
            and
            name.isdigit()
        ):

            numbers.append(
                int(name)
            )

    if not numbers:
        return 1

    return max(
        numbers
    ) + 1

def save_results_transactional(
    results,
    output_number
):

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    final_folder = os.path.join(
        OUTPUT_ROOT,
        str(output_number)
    )

    if os.path.exists(
        final_folder
    ):

        print()
        print(
            "[ERROR] 출력 폴더가 이미 존재합니다."
        )

        print(
            final_folder
        )

        return False

    temp_folder = tempfile.mkdtemp(
        prefix=f".tmp_{output_number}_",
        dir=OUTPUT_ROOT
    )

    try:

        count = 0

        for number, image in enumerate(
            results,
            start=1
        ):

            path = os.path.join(
                temp_folder,
                f"original_{number}.png"
            )

            success = imwrite_korean(
                path,
                image,
                ".png"
            )

            if not success:

                raise RuntimeError(
                    f"{number}.png 저장 실패"
                )

            height, width = (
                image.shape[:2]
            )

            print(
                f"    original_{number}.png "
                f"| {width} x {height}"
            )

            count += 1

        if count != CAM0_ROI_COUNT + CAM1_ROI_COUNT:

            raise RuntimeError(
                f"저장 개수 오류: {count}"
            )

        os.rename(
            temp_folder,
            final_folder
        )

        return True

    except Exception as e:

        print()
        print("[ERROR] ROI 저장 실패")
        print(e)

        if os.path.isdir(
            temp_folder
        ):

            shutil.rmtree(
                temp_folder,
                ignore_errors=True
            )

        return False

def prepare_full_resolution_maps(
    distortion_config
):

    values0 = (
        get_camera_distortion_values(
            distortion_config,
            0
        )
    )

    values1 = (
        get_camera_distortion_values(
            distortion_config,
            1
        )
    )

    cam0_map1, cam0_map2 = (
        create_undistort_map(
            EXPECTED_WIDTH,
            EXPECTED_HEIGHT,
            values0
        )
    )

    cam1_map1, cam1_map2 = (
        create_undistort_map(
            EXPECTED_WIDTH,
            EXPECTED_HEIGHT,
            values1
        )
    )

    return {

        "cam0_map1":
            cam0_map1,

        "cam0_map2":
            cam0_map2,

        "cam1_map1":
            cam1_map1,

        "cam1_map2":
            cam1_map2
    }


def run_batch():

    input_sets = (
        discover_input_sets()
    )

    if not input_sets:

        print()
        print(
            "[ERROR] 처리할 input 데이터가 없습니다."
        )

        return

    distortion_config = (
        get_current_distortion_config()
    )

    roi_config = (
        get_current_roi_config()
    )

    if not validate_roi_config(
        roi_config
    ):

        print()
        print(
            "[ERROR] 현재 ROI 좌표가 잘못되었습니다."
        )

        return

    print()
    print("============================================")
    print("Full Resolution Lens Map 생성")
    print("============================================")

    maps = (
        prepare_full_resolution_maps(
            distortion_config
        )
    )

    print("완료")


    next_output = (
        get_next_output_number()
    )

    success_count = 0
    fail_count = 0


    for source_name, folder_path in input_sets:

        print()
        print("============================================")

        print(
            "처리:",
            source_name
        )

        print("============================================")


        cam0, cam1 = load_camera_images(
            folder_path
        )

        if (
            cam0 is None
            or
            cam1 is None
        ):

            fail_count += 1
            continue


        if not check_resolution(
            cam0,
            "CAM0"
        ):

            fail_count += 1
            continue

        if not check_resolution(
            cam1,
            "CAM1"
        ):

            fail_count += 1
            continue

        corrected0 = apply_undistort(cam0, maps["cam0_map1"], maps["cam0_map2"])
        corrected0 = rotate_with_config(corrected0, distortion_config["CAM0_ROTATION"])

        corrected1 = apply_undistort(cam1, maps["cam1_map1"], maps["cam1_map2"])
        corrected1 = rotate_with_config(corrected1, distortion_config["CAM1_ROTATION"])

        results = extract_rois(
            corrected0,
            corrected1,
            roi_config
        )

        if results is None:

            print()
            print(
                "[ERROR] ROI 추출 실패"
            )

            fail_count += 1
            continue

        success = save_results_transactional(
            results,
            next_output
        )

        if success:

            print()
            print(
                f"[SUCCESS] "
                f"{source_name} "
                f"-> output/{next_output}"
            )

            next_output += 1
            success_count += 1

        else:

            fail_count += 1


    print()
    print("============================================")
    print("전체 처리 완료")
    print("============================================")

    print(
        "성공:",
        success_count
    )

    print(
        "실패/건너뜀:",
        fail_count
    )

    print(
        "출력:",
        OUTPUT_ROOT
    )

    print("============================================")

def main():
    load_active_profile()

    print()
    print("============================================")
    print("Dual Camera Signature ROI")
    print("============================================")

    print(
        "RUN_MODE:",
        RUN_MODE
    )

    print(
        "INPUT:",
        INPUT_ROOT
    )

    print(
        "OUTPUT:",
        OUTPUT_ROOT
    )

    print()
    print(f"ROI GRID: CAM0={CAM0_ROI_COLS}x{CAM0_ROI_ROWS} ({CAM0_ROI_COUNT}개), CAM1={CAM1_ROI_COLS}x{CAM1_ROI_ROWS} ({CAM1_ROI_COUNT}개), 총={CAM0_ROI_COUNT + CAM1_ROI_COUNT}개")
    print(f"프로파일: {get_profile_path()}")
    print("현재 초기 렌즈값")

    print(
        f"CAM0: "
        f"K1={CAM0_K1:+.4f}, "
        f"K2={CAM0_K2:+.4f}, "
        f"K3={CAM0_K3:+.4f}"
    )

    print(
        f"CAM1: "
        f"K1={CAM1_K1:+.4f}, "
        f"K2={CAM1_K2:+.4f}, "
        f"K3={CAM1_K3:+.4f}"
    )

    print("============================================")


    if RUN_MODE == "calibration":

        run_calibration()


    elif RUN_MODE == "batch":

        run_batch()


    else:

        print()
        print(
            "[ERROR] RUN_MODE가 잘못되었습니다."
        )

        print()
        print(
            'RUN_MODE = "calibration"'
        )

        print(
            'RUN_MODE = "batch"'
        )


if __name__ == "__main__":

    main()
