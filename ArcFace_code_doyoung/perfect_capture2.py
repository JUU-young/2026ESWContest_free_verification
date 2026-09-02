import os
import shutil
import time
import threading
import subprocess
import traceback
from datetime import datetime

import cv2
from flask import Flask, Response

try:
    import Jetson.GPIO as GPIO
except ImportError:
    GPIO = None


USE_GPIO_LIGHT = True
LIGHT_RELAY_PIN = 11
LIGHT_ACTIVE_HIGH = True
LIGHT_WARMUP_SECONDS = 0.5

CAMERA_1_INDEX = 0
CAMERA_2_INDEX = 2

CAMERA_1_DEVICE = "/dev/video0"
CAMERA_2_DEVICE = "/dev/video2"

CAMERA_1_FILENAME_PREFIX = "cam1"
CAMERA_2_FILENAME_PREFIX = "cam0"

CAM_WIDTH = 3264
CAM_HEIGHT = 2448
CAM_FPS = 15

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
PREVIEW_JPEG_QUALITY = 60

SAVE_JPEG_QUALITY = 95


SHUTTER_VALUE = 167

CAMERA_1_GAIN = 10
CAMERA_2_GAIN = 10

CAMERA_1_BRIGHTNESS = 0
CAMERA_2_BRIGHTNESS = 0


BASE_SAVE_DIR = "/home/lab/signature_capture"

os.makedirs(
    BASE_SAVE_DIR,
    exist_ok=True
)


HOST = "0.0.0.0"
PORT = 5000

app = Flask(__name__)


running = True

cv2.setNumThreads(1)

save_sequence_lock = threading.Lock()
light_lock = threading.Lock()

light_initialized = False
light_is_on = False


def setup_light_gpio():
    global light_initialized
    global light_is_on

    if not USE_GPIO_LIGHT:
        print("[GPIO] 조명 기능 비활성화")
        return True

    if GPIO is None:
        print()
        print("[GPIO ERROR] Jetson.GPIO 모듈을 불러올 수 없습니다.")
        print("카메라 기능은 계속 실행합니다.")
        return False

    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)

        off_level = (
            GPIO.LOW
            if LIGHT_ACTIVE_HIGH
            else GPIO.HIGH
        )

        GPIO.setup(
            LIGHT_RELAY_PIN,
            GPIO.OUT,
            initial=off_level
        )

        light_initialized = True
        light_is_on = False

        print()
        print("==============================================")
        print("GPIO LIGHT READY")
        print("Pin         :", LIGHT_RELAY_PIN)
        print("Mode        : BOARD")
        print("Active High :", LIGHT_ACTIVE_HIGH)
        print("Initial     : OFF")
        print("==============================================")

        return True

    except Exception as e:
        print()
        print("[GPIO ERROR] 조명 초기화 실패:", repr(e))
        light_initialized = False
        light_is_on = False
        return False

def set_light(on):
    global light_is_on

    if not USE_GPIO_LIGHT:
        return True

    if GPIO is None or not light_initialized:
        return False

    with light_lock:
        try:
            if LIGHT_ACTIVE_HIGH:
                level = GPIO.HIGH if on else GPIO.LOW
            else:
                level = GPIO.LOW if on else GPIO.HIGH

            GPIO.output(
                LIGHT_RELAY_PIN,
                level
            )

            light_is_on = bool(on)

            print(
                "[LIGHT]",
                "ON" if on else "OFF"
            )

            return True

        except Exception as e:
            print(
                "[GPIO ERROR] 조명 제어 실패:",
                repr(e)
            )
            return False


def light_on():
    return set_light(True)


def light_off():
    return set_light(False)


def cleanup_light_gpio():
    global light_initialized
    global light_is_on

    if GPIO is None or not light_initialized:
        return

    try:
        light_off()
        GPIO.cleanup(LIGHT_RELAY_PIN)
        print("[GPIO] Light GPIO cleanup complete.")

    except Exception as e:
        print(
            "[GPIO WARNING] cleanup 실패:",
            repr(e)
        )

    finally:
        light_initialized = False
        light_is_on = False

def set_camera_controls(device, gain, brightness):
    print()
    print("==============================================")
    print("[CAMERA SETTING]", device)
    print("==============================================")

    controls = [
        "exposure_auto=1",
        f"exposure_absolute={SHUTTER_VALUE}",
        f"gain={gain}",
        f"brightness={brightness}",
    ]

    try:
        for control in controls:
            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    device,
                    "-c",
                    control,
                ],
                check=True
            )

        print("[OK]", device)
        print("Shutter    : 1/60 sec")
        print("Exposure   :", SHUTTER_VALUE)
        print("Gain       :", gain)
        print("Brightness :", brightness)

    except subprocess.CalledProcessError as e:
        print()
        print(
            f"[ERROR] {device} camera setting failed"
        )
        print(repr(e))

class CameraWorker:
    def __init__(
        self,
        index,
        device,
        name,
        gain,
        brightness
    ):
        self.index = index
        self.device = device
        self.name = name
        self.gain = gain
        self.brightness = brightness

        self.cap = None

        self.latest_original_frame = None
        self.latest_jpeg = None
        self.latest_frame_id = 0

        self.running = True

        self.frame_lock = threading.Lock()
        self.frame_condition = threading.Condition(
            self.frame_lock
        )

        self.thread = threading.Thread(
            target=self.camera_loop,
            daemon=True
        )

    def start(self):
        self.thread.start()

    def camera_loop(self):
        print()
        print(
            f"Opening {self.name}: {self.device}"
        )

        self.cap = cv2.VideoCapture(
            self.index,
            cv2.CAP_V4L2
        )

        if not self.cap.isOpened():
            print()
            print(
                f"[ERROR] {self.name} could not be opened."
            )
            print("Device:", self.device)
            self.running = False
            return

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAM_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAM_HEIGHT
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            CAM_FPS
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        actual_width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        actual_height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        actual_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        fourcc = int(
            self.cap.get(cv2.CAP_PROP_FOURCC)
        )

        fourcc_text = "".join(
            [
                chr(fourcc & 0xFF),
                chr((fourcc >> 8) & 0xFF),
                chr((fourcc >> 16) & 0xFF),
                chr((fourcc >> 24) & 0xFF),
            ]
        )

        print()
        print("==============================================")
        print(f"{self.name} OPEN SUCCESS")
        print("Device     :", self.device)
        print(
            "Resolution :",
            actual_width,
            "x",
            actual_height
        )
        print("FPS        :", actual_fps)
        print("FOURCC     :", fourcc_text)
        print("Shutter    : 1/60 sec")
        print("Exposure   :", SHUTTER_VALUE)
        print("Gain       :", self.gain)
        print("Brightness :", self.brightness)
        print("==============================================")
        print()

        while running and self.running:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                print(
                    f"[WARNING] {self.name} frame read failed"
                )
                time.sleep(0.01)
                continue

            try:
                preview = cv2.resize(
                    frame,
                    (
                        PREVIEW_WIDTH,
                        PREVIEW_HEIGHT
                    ),
                    interpolation=cv2.INTER_AREA
                )

                success, jpeg = cv2.imencode(
                    ".jpg",
                    preview,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        PREVIEW_JPEG_QUALITY
                    ]
                )

                if not success:
                    continue

                jpeg_bytes = jpeg.tobytes()

                with self.frame_condition:
                    self.latest_original_frame = frame
                    self.latest_jpeg = jpeg_bytes
                    self.latest_frame_id += 1
                    self.frame_condition.notify_all()

            except Exception as e:
                print(
                    f"[{self.name} ERROR]",
                    repr(e)
                )
                traceback.print_exc()
                time.sleep(0.05)

        if self.cap is not None:
            self.cap.release()

        print(
            f"{self.name} released."
        )

    def get_frame_id(self):
        with self.frame_lock:
            return self.latest_frame_id

    def wait_for_frame_after(
        self,
        previous_frame_id,
        timeout=2.0
    ):
        end_time = time.monotonic() + timeout

        with self.frame_condition:
            while (
                running
                and self.running
                and self.latest_frame_id <= previous_frame_id
            ):
                remaining = end_time - time.monotonic()

                if remaining <= 0:
                    return False

                self.frame_condition.wait(
                    timeout=remaining
                )

            return (
                self.latest_original_frame is not None
                and self.latest_frame_id > previous_frame_id
            )

    def get_original_frame(self):
        with self.frame_lock:
            if self.latest_original_frame is None:
                return None

            return self.latest_original_frame.copy()

    def generate_stream(self):
        last_frame_id = -1

        try:
            while running and self.running:
                with self.frame_condition:
                    self.frame_condition.wait_for(
                        lambda: (
                            not running
                            or not self.running
                            or (
                                self.latest_jpeg is not None
                                and self.latest_frame_id != last_frame_id
                            )
                        ),
                        timeout=1.0
                    )

                    if not running or not self.running:
                        break

                    if self.latest_jpeg is None:
                        continue

                    if self.latest_frame_id == last_frame_id:
                        continue

                    frame = self.latest_jpeg
                    last_frame_id = self.latest_frame_id

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n"
                    b"\r\n"
                    + frame
                    + b"\r\n"
                )

        except GeneratorExit:
            return

        except Exception as e:
            print(
                f"[{self.name} STREAM ERROR]",
                repr(e)
            )
            return

    def stop(self):
        self.running = False

        with self.frame_condition:
            self.frame_condition.notify_all()

        if self.thread.is_alive():
            self.thread.join(timeout=2)

        if (
            self.cap is not None
            and self.cap.isOpened()
        ):
            self.cap.release()


camera1 = CameraWorker(
    CAMERA_1_INDEX,
    CAMERA_1_DEVICE,
    "Camera 1",
    CAMERA_1_GAIN,
    CAMERA_1_BRIGHTNESS
)

camera2 = CameraWorker(
    CAMERA_2_INDEX,
    CAMERA_2_DEVICE,
    "Camera 2",
    CAMERA_2_GAIN,
    CAMERA_2_BRIGHTNESS
)


# ============================================================
# 다음 촬영 폴더 번호
# ============================================================

def get_next_capture_number():
    os.makedirs(
        BASE_SAVE_DIR,
        exist_ok=True
    )

    numbers = []

    for name in os.listdir(
        BASE_SAVE_DIR
    ):
        path = os.path.join(
            BASE_SAVE_DIR,
            name
        )

        if (
            os.path.isdir(path)
            and name.isdigit()
        ):
            numbers.append(int(name))

    if not numbers:
        return 1

    return max(numbers) + 1


# ============================================================
# JPEG 저장
# ============================================================

def save_frame_as_jpeg(
    frame,
    filepath,
    camera_name,
    gain,
    brightness
):
    if frame is None:
        print(
            f"[SAVE ERROR] {camera_name}: frame 없음"
        )
        return False

    start = time.perf_counter()

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            SAVE_JPEG_QUALITY
        ]
    )

    if not success:
        print(
            f"[SAVE ERROR] {camera_name}: JPEG encoding 실패"
        )
        return False

    try:
        with open(filepath, "wb") as file:
            file.write(encoded.tobytes())

    except Exception as e:
        print(
            f"[SAVE ERROR] {camera_name}:",
            repr(e)
        )
        return False

    elapsed = time.perf_counter() - start
    height, width = frame.shape[:2]
    file_size = os.path.getsize(filepath)

    print()
    print("==============================================")
    print(f"[{camera_name} HIGH QUALITY SAVED]")
    print("File       :", filepath)
    print("Resolution :", f"{width} x {height}")
    print("JPEG       :", SAVE_JPEG_QUALITY)
    print("Shutter    : 1/60")
    print("Gain       :", gain)
    print("Brightness :", brightness)
    print(
        "File Size  :",
        f"{file_size / 1024 / 1024:.2f} MB"
    )
    print(
        "Save Time  :",
        f"{elapsed:.4f} sec"
    )
    print("==============================================")
    print()

    return True


# ============================================================
# 두 카메라 한 세트 저장
#
# Enter 1회:
#
# 1/
#   cam1_....jpg <- Camera 1
#   cam0_....jpg <- Camera 2
#
# 촬영 과정:
# 1. 조명 ON
# 2. 0.6초 대기
# 3. 조명 ON 이후 새 프레임 확인
# 4. 두 카메라 프레임 복사
# 5. 같은 폴더에 저장
# 6. 조명 OFF
# ============================================================

def save_both_cameras():
    with save_sequence_lock:
        temp_folder = None

        try:
            print()
            print("==============================================")
            print("HIGH QUALITY CAPTURE START")
            print("Camera 1(cam1) + Camera 2(cam0)")
            print("==============================================")

            # 조명 켜기 전 프레임 번호 기억
            before_id_1 = camera1.get_frame_id()
            before_id_2 = camera2.get_frame_id()

            # =================================================
            # GPIO 조명 ON
            # =================================================

            if USE_GPIO_LIGHT:
                if not light_on():
                    print(
                        "[WARNING] 조명을 켜지 못했습니다. "
                        "촬영은 계속합니다."
                    )
                else:
                    time.sleep(
                        LIGHT_WARMUP_SECONDS
                    )

                    # 조명이 켜진 뒤의 새 프레임을 기다림
                    camera1.wait_for_frame_after(
                        before_id_1,
                        timeout=2.0
                    )

                    camera2.wait_for_frame_after(
                        before_id_2,
                        timeout=2.0
                    )

            # =================================================
            # 두 카메라의 최신 고해상도 프레임을 먼저 복사
            # =================================================

            frame1 = camera1.get_original_frame()
            frame2 = camera2.get_original_frame()

            if frame1 is None or frame2 is None:
                raise RuntimeError(
                    "두 카메라 중 하나의 최신 프레임이 없습니다."
                )

            # =================================================
            # 이번 촬영 폴더
            # =================================================

            capture_number = get_next_capture_number()

            final_folder = os.path.join(
                BASE_SAVE_DIR,
                str(capture_number)
            )

            temp_folder = os.path.join(
                BASE_SAVE_DIR,
                f".tmp_{capture_number}"
            )

            if os.path.exists(temp_folder):
                shutil.rmtree(
                    temp_folder,
                    ignore_errors=True
                )

            os.makedirs(
                temp_folder,
                exist_ok=False
            )

            capture_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )[:-3]

            cam1_filename = (
                f"{CAMERA_1_FILENAME_PREFIX}_{capture_id}.jpg"
            )

            cam0_filename = (
                f"{CAMERA_2_FILENAME_PREFIX}_{capture_id}.jpg"
            )

            cam1_path = os.path.join(
                temp_folder,
                cam1_filename
            )

            cam0_path = os.path.join(
                temp_folder,
                cam0_filename
            )

            print()
            print("Capture Folder:", capture_number)

            # =================================================
            # Camera 1 -> cam1
            # =================================================

            camera1_success = save_frame_as_jpeg(
                frame1,
                cam1_path,
                "Camera 1 -> cam1",
                CAMERA_1_GAIN,
                CAMERA_1_BRIGHTNESS
            )

            # =================================================
            # Camera 2 -> cam0
            # =================================================

            camera2_success = save_frame_as_jpeg(
                frame2,
                cam0_path,
                "Camera 2 -> cam0",
                CAMERA_2_GAIN,
                CAMERA_2_BRIGHTNESS
            )

            if not (
                camera1_success
                and camera2_success
            ):
                raise RuntimeError(
                    "두 이미지 중 하나 이상 저장 실패"
                )

            if os.path.exists(final_folder):
                raise RuntimeError(
                    f"최종 폴더가 이미 존재합니다: {final_folder}"
                )

            # 두 장 모두 성공했을 때만 최종 폴더 확정
            os.rename(
                temp_folder,
                final_folder
            )

            temp_folder = None

            print()
            print("==============================================")
            print("CAPTURE COMPLETE")
            print("Folder:", final_folder)
            print("Saved Pair:")
            print(
                f"  {capture_number}/{cam1_filename}"
            )
            print(
                f"  {capture_number}/{cam0_filename}"
            )
            print("==============================================")
            print()

            return True

        except Exception as e:
            print()
            print("==============================================")
            print("CAPTURE FAILED")
            print(repr(e))
            print("불완전한 촬영 세트는 저장하지 않습니다.")
            print("==============================================")
            traceback.print_exc()

            if (
                temp_folder is not None
                and os.path.isdir(temp_folder)
            ):
                shutil.rmtree(
                    temp_folder,
                    ignore_errors=True
                )

            return False

        finally:
            # 성공/실패와 무관하게 촬영 후 조명 OFF
            if USE_GPIO_LIGHT:
                light_off()


# ============================================================
# Terminal
# ============================================================

def terminal_input_worker():
    print()
    print("Waiting for both cameras...")

    while running:
        camera1_ready = (
            camera1.get_original_frame() is not None
        )

        camera2_ready = (
            camera2.get_original_frame() is not None
        )

        if camera1_ready and camera2_ready:
            break

        time.sleep(0.1)

    if not running:
        return

    print()
    print("==============================================")
    print("DUAL CAMERA READY")
    print()
    print("Camera 1 -> cam1")
    print("Camera 2 -> cam0")
    print()
    print("GPIO Light Pin:", LIGHT_RELAY_PIN)
    print("Enter:")
    print("조명 ON -> 두 카메라 촬영 -> 조명 OFF")
    print("Ctrl+C: 종료")
    print("==============================================")
    print()

    while running:
        try:
            input(
                "고화질 사진 저장 Enter > "
            )

            if not running:
                break

            save_both_cameras()

        except EOFError:
            break

        except Exception as e:
            if USE_GPIO_LIGHT:
                light_off()

            print(
                "[TERMINAL ERROR]",
                repr(e)
            )
            traceback.print_exc()


# ============================================================
# Flask 화면
# ============================================================

@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dual Camera</title>
<style>
html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    background: #111;
    color: white;
    overflow: hidden;
    font-family: Arial, sans-serif;
}
.header {
    height: 60px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 22px;
}
.container {
    width: 100%;
    height: calc(100vh - 60px);
    display: flex;
}
.camera-box {
    width: 50%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 5px;
    box-sizing: border-box;
}
.camera-title {
    height: 40px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 18px;
}
.camera-image {
    width: 100%;
    height: calc(100% - 40px);
    object-fit: contain;
    border: 1px solid #444;
    box-sizing: border-box;
}
</style>
</head>
<body>
<div class="header">
3264 × 2448 &nbsp; | &nbsp; 15 FPS &nbsp; | &nbsp; Shutter 1/60
</div>
<div class="container">
    <div class="camera-box">
        <div class="camera-title">Camera 2 → cam0</div>
        <img class="camera-image" src="/camera2">
    </div>
    <div class="camera-box">
        <div class="camera-title">Camera 1 → cam1</div>
        <img class="camera-image" src="/camera1">
    </div>
</div>
</body>
</html>
"""


# ============================================================
# Camera 1 Stream
# ============================================================

@app.route("/camera1")
def camera1_feed():
    response = Response(
        camera1.generate_stream(),
        mimetype=(
            "multipart/x-mixed-replace;"
            "boundary=frame"
        )
    )

    response.headers["Cache-Control"] = (
        "no-cache, no-store, must-revalidate"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


# ============================================================
# Camera 2 Stream
# ============================================================

@app.route("/camera2")
def camera2_feed():
    response = Response(
        camera2.generate_stream(),
        mimetype=(
            "multipart/x-mixed-replace;"
            "boundary=frame"
        )
    )

    response.headers["Cache-Control"] = (
        "no-cache, no-store, must-revalidate"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print()
    print("==============================================")
    print("HIGH QUALITY DUAL CAMERA + GPIO LIGHT")
    print("==============================================")
    print("Camera 1:", CAMERA_1_DEVICE, "-> cam1")
    print("  Brightness:", CAMERA_1_BRIGHTNESS)
    print("  Gain      :", CAMERA_1_GAIN)
    print()
    print("Camera 2:", CAMERA_2_DEVICE, "-> cam0")
    print("  Brightness:", CAMERA_2_BRIGHTNESS)
    print("  Gain      :", CAMERA_2_GAIN)
    print()
    print("Resolution :", f"{CAM_WIDTH} x {CAM_HEIGHT}")
    print("FPS        :", CAM_FPS)
    print("Shutter    : 1/60 sec")
    print("JPEG       :", SAVE_JPEG_QUALITY)
    print("GPIO Pin   :", LIGHT_RELAY_PIN)
    print("Light Mode : Capture ON -> Save -> OFF")
    print("Save Root  :", BASE_SAVE_DIR)
    print("==============================================")

    # GPIO 먼저 초기화
    setup_light_gpio()

    # 카메라 controls
    set_camera_controls(
        CAMERA_1_DEVICE,
        CAMERA_1_GAIN,
        CAMERA_1_BRIGHTNESS
    )

    set_camera_controls(
        CAMERA_2_DEVICE,
        CAMERA_2_GAIN,
        CAMERA_2_BRIGHTNESS
    )

    # 카메라 시작
    camera1.start()

    time.sleep(0.3)

    camera2.start()

    # Terminal thread
    terminal_thread = threading.Thread(
        target=terminal_input_worker,
        daemon=True
    )
    terminal_thread.start()

    print()
    print("==============================================")
    print("Flask Server")
    print(f"http://Jetson_IP:{PORT}")
    print()
    print("Save structure:")
    print("  1/cam1_...jpg  <- Camera 1")
    print("  1/cam0_...jpg  <- Camera 2")
    print("  2/cam1_...jpg + cam0_...jpg")
    print("==============================================")
    print()

    try:
        app.run(
            host=HOST,
            port=PORT,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    finally:
        running = False

        print()
        print("Stopping cameras...")

        camera1.stop()
        camera2.stop()

        cleanup_light_gpio()

        print("Program finished.")
