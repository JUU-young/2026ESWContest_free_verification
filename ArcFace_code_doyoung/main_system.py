import shutil
import time
import select
from pathlib import Path

from evdev import InputDevice, list_devices, ecodes
from RPLCD.i2c import CharLCD

import perfect_capture2 as capture
import ROI_final_grid_profile as roi

from signature_registry_sqlite import (
    get_device,
    load_model,
    enroll_user,
    verify_user,
    connect_database,
)

from yolo_best_roi import (
    load_yolo_model,
    extract_best_signature,
)

BASE_DIR = Path(__file__).resolve().parent

RUNTIME_DIR = BASE_DIR / "runtime"
REGISTER_ROI_DIR = RUNTIME_DIR / "register_rois"
VERIFY_ROI_PATH = RUNTIME_DIR / "verify_signature.png"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

LCD_I2C_BUS = 7
LCD_ADDRESS = 0x27

LCD_COLS = 16
LCD_ROWS = 2

NOT_REGISTERED_SECONDS = 2.0
COMPLETE_SECONDS = 3.0
RESULT_SECONDS = 3.0
ERROR_SECONDS = 2.0

lcd = None
keyboard = None

device = None
backbone = None

yolo_model = None
yolo_device = None

roi_distortion_config = None
roi_config = None
roi_maps = None

current_mode = "waiting"
current_id = None

def init_lcd():
    global lcd

    lcd = CharLCD(
        i2c_expander="PCF8574",
        address=LCD_ADDRESS,
        port=LCD_I2C_BUS,
        cols=LCD_COLS,
        rows=LCD_ROWS,
        charmap="A00",
        auto_linebreaks=False,
    )

    lcd.clear()
    lcd.cursor_mode = "hide"


def lcd_show(line1="", line2="", cursor=False, cursor_position=None):
    line1 = str(line1)[:LCD_COLS]
    line2 = str(line2)[:LCD_COLS]

    lcd.cursor_mode = "hide"

    lcd.cursor_pos = (0, 0)
    lcd.write_string(line1.ljust(LCD_COLS))

    lcd.cursor_pos = (1, 0)
    lcd.write_string(line2.ljust(LCD_COLS))

    if cursor:
        if cursor_position is None:
            cursor_position = (1, min(len(line2), LCD_COLS - 1))

        lcd.cursor_pos = cursor_position
        lcd.cursor_mode = "blink"


def show_waiting():
    global current_mode, current_id

    current_mode = "waiting"
    current_id = None

    lcd_show(
        "Waiting",
        "L:REG U:DEL R:V"
    )

    print()
    print("[WAITING]")
    print("LEFT  = registration")
    print("UP    = delete user")
    print("RIGHT = verification")
    print("DOWN  = capture (ready 상태에서)")

def find_usb_keyboard():
    print()
    print("[KEYBOARD] Searching...")

    target_path = "/dev/input/event0"

    try:
        dev = InputDevice(target_path)

        print(
            f"[KEYBOARD CHECK] "
            f"{dev.name} {dev.path}"
        )

        caps = dev.capabilities()

        if ecodes.EV_KEY not in caps:
            print("[KEYBOARD ERROR] EV_KEY 없음")
            return None

        keys = caps[ecodes.EV_KEY]

        required_keys = [
        
            ecodes.KEY_LEFT,
            ecodes.KEY_RIGHT,
            ecodes.KEY_UP,
            ecodes.KEY_DOWN,

            # 제어
            ecodes.KEY_ENTER,
            ecodes.KEY_ESC,
            ecodes.KEY_BACKSPACE,

            # 숫자
            ecodes.KEY_0,

            # 영어
            ecodes.KEY_A,
            ecodes.KEY_Z,
        ]

        missing = [
            ecodes.KEY.get(key, str(key))
            for key in required_keys
            if key not in keys
        ]

        if missing:
            print(
                "[KEYBOARD ERROR] "
                f"필수 키 없음: {missing}"
            )
            return None

        print(
            f"[KEYBOARD 발견!] "
            f"{dev.name} {dev.path}"
        )

        return dev

    except Exception as e:
        print(
            f"[KEYBOARD ERROR] "
            f"{target_path}: {e}"
        )
        return None

ID_KEYS = {
    # 숫자열
    ecodes.KEY_0: "0",
    ecodes.KEY_1: "1",
    ecodes.KEY_2: "2",
    ecodes.KEY_3: "3",
    ecodes.KEY_4: "4",
    ecodes.KEY_5: "5",
    ecodes.KEY_6: "6",
    ecodes.KEY_7: "7",
    ecodes.KEY_8: "8",
    ecodes.KEY_9: "9",

    # 영문
    ecodes.KEY_A: "A",
    ecodes.KEY_B: "B",
    ecodes.KEY_C: "C",
    ecodes.KEY_D: "D",
    ecodes.KEY_E: "E",
    ecodes.KEY_F: "F",
    ecodes.KEY_G: "G",
    ecodes.KEY_H: "H",
    ecodes.KEY_I: "I",
    ecodes.KEY_J: "J",
    ecodes.KEY_K: "K",
    ecodes.KEY_L: "L",
    ecodes.KEY_M: "M",
    ecodes.KEY_N: "N",
    ecodes.KEY_O: "O",
    ecodes.KEY_P: "P",
    ecodes.KEY_Q: "Q",
    ecodes.KEY_R: "R",
    ecodes.KEY_S: "S",
    ecodes.KEY_T: "T",
    ecodes.KEY_U: "U",
    ecodes.KEY_V: "V",
    ecodes.KEY_W: "W",
    ecodes.KEY_X: "X",
    ecodes.KEY_Y: "Y",
    ecodes.KEY_Z: "Z",

   
    ecodes.KEY_KP0: "0",
    ecodes.KEY_KP1: "1",
    ecodes.KEY_KP2: "2",
    ecodes.KEY_KP3: "3",
    ecodes.KEY_KP4: "4",
    ecodes.KEY_KP5: "5",
    ecodes.KEY_KP6: "6",
    ecodes.KEY_KP7: "7",
    ecodes.KEY_KP8: "8",
    ecodes.KEY_KP9: "9",
}

def read_keypress():
    while True:
        select.select([keyboard.fd], [], [])

        for event in keyboard.read():
            if event.type != ecodes.EV_KEY:
                continue

            if event.value != 1:
                continue

            return event.code


def read_id_from_keyboard():
    text = ""

    lcd_show(
        "ID:",
        "",
        cursor=True,
        cursor_position=(1, 0),
    )

    print("[ID] 숫자 입력 후 Enter")
    print("[ID] ESC = 취소")

    while True:
        code = read_keypress()

        if code == ecodes.KEY_ESC:
            lcd.cursor_mode = "hide"
            return None

        if code in (ecodes.KEY_ENTER, ecodes.KEY_KPENTER):
            if text:
                lcd.cursor_mode = "hide"
                print(f"[ID] entered: {text}")
                return text
            continue

        if code == ecodes.KEY_BACKSPACE:
            if text:
                text = text[:-1]

        elif code in ID_KEYS:
            if len(text) < LCD_COLS:
                text += ID_KEYS[code]

        else:
            continue

        lcd_show(
            "ID:",
            text,
            cursor=True,
            cursor_position=(1, min(len(text), LCD_COLS - 1)),
        )


def is_registered_id(user_id):
    conn = connect_database()

    try:
        row = conn.execute(
            """
            SELECT 1
            FROM users
            WHERE id = ?
            LIMIT 1
            """,
            (str(user_id),),
        ).fetchone()

    finally:
        conn.close()

    return row is not None


def delete_user_from_db(user_id):

    conn = connect_database()

    try:

        cursor = conn.execute(
            """
            DELETE FROM users
            WHERE id = ?
            """,
            (
                str(user_id),
            )
        )

        conn.commit()

        return (
            cursor.rowcount
            >
            0
        )

    finally:

        conn.close()


def enter_delete_mode():

    global current_mode

    current_mode = "delete_id"

    lcd_show(
        "Delete user",
        "Input ID"
    )

    time.sleep(0.3)

    user_id = read_id_from_keyboard()

    if user_id is None:

        show_waiting()
        return

    if not is_registered_id(
        user_id
    ):

        lcd_show(
            "Not registered",
            ""
        )

        print(
            f"[DELETE] 등록되지 않은 ID: "
            f"{user_id}"
        )

        time.sleep(
            NOT_REGISTERED_SECONDS
        )

        show_waiting()
        return


    lcd_show(
        "Delete ID?",
        f"{user_id} ENTER"
    )

    print()
    print(
        f"[DELETE] ID={user_id}"
    )
    print(
        "[DELETE] Enter = 삭제 / ESC = 취소"
    )

    while True:

        code = read_keypress()

        if code in (
            ecodes.KEY_ENTER,
            ecodes.KEY_KPENTER
        ):

            deleted = (
                delete_user_from_db(
                    user_id
                )
            )

            if deleted:

                lcd_show(
                    "Delete complete",
                    str(user_id)
                )

                print(
                    f"[DELETE] ID {user_id} "
                    f"삭제 완료"
                )

                time.sleep(
                    COMPLETE_SECONDS
                )

            else:

                lcd_show(
                    "Delete error",
                    ""
                )

                print(
                    f"[DELETE ERROR] "
                    f"ID {user_id}"
                )

                time.sleep(
                    ERROR_SECONDS
                )

            show_waiting()
            return

        elif code == ecodes.KEY_ESC:

            lcd_show(
                "Delete cancel",
                ""
            )

            print(
                "[DELETE] 취소"
            )

            time.sleep(
                1.0
            )

            show_waiting()
            return


def init_cameras():
    capture.setup_light_gpio()

    capture.set_camera_controls(
        capture.CAMERA_1_DEVICE,
        capture.CAMERA_1_GAIN,
        capture.CAMERA_1_BRIGHTNESS,
    )

    capture.set_camera_controls(
        capture.CAMERA_2_DEVICE,
        capture.CAMERA_2_GAIN,
        capture.CAMERA_2_BRIGHTNESS,
    )

    capture.camera1.start()
    time.sleep(0.3)
    capture.camera2.start()

    print("[CAMERA] Waiting for camera frames...")

    deadline = time.monotonic() + 15.0

    while time.monotonic() < deadline:
        ready1 = capture.camera1.get_original_frame() is not None
        ready2 = capture.camera2.get_original_frame() is not None

        if ready1 and ready2:
            print("[CAMERA] Both cameras ready.")
            return

        time.sleep(0.1)

    raise RuntimeError("카메라 2대가 준비되지 않았습니다.")


def capture_pair():
    capture_number = capture.get_next_capture_number()

    success = capture.save_both_cameras()

    if not success:
        return None

    folder = Path(capture.BASE_SAVE_DIR) / str(capture_number)

    if not folder.is_dir():
        return None

    return folder


def init_registration_roi():
    global roi_distortion_config, roi_config, roi_maps

    roi.load_active_profile()

    roi_distortion_config = roi.get_current_distortion_config()
    roi_config = roi.get_current_roi_config()

    if not roi.validate_roi_config(roi_config):
        raise RuntimeError("ROI 설정이 올바르지 않습니다.")

    roi_maps = roi.prepare_full_resolution_maps(
        roi_distortion_config
    )

    expected = roi.CAM0_ROI_COUNT + roi.CAM1_ROI_COUNT

    print(f"[ROI] Registration ROI count: {expected}")

    if expected != 14:
        raise RuntimeError(
            f"등록 ROI 개수가 14가 아닙니다: {expected}"
        )


def make_registration_rois(capture_folder):
    if REGISTER_ROI_DIR.exists():
        shutil.rmtree(
            REGISTER_ROI_DIR,
            ignore_errors=True
        )

    REGISTER_ROI_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cam0, cam1 = roi.load_camera_images(
        str(capture_folder),
        verbose=False
    )
    cam0, cam1 = cam1, cam0
    
    if cam0 is None or cam1 is None:
        raise RuntimeError(
            "cam0/cam1 촬영 이미지를 찾지 못했습니다."
        )

    if not roi.check_resolution(
        cam0,
        "CAM0",
        verbose=False
    ):
        raise RuntimeError("CAM0 해상도 오류")

    if not roi.check_resolution(
        cam1,
        "CAM1",
        verbose=False
    ):
        raise RuntimeError("CAM1 해상도 오류")

    corrected0 = roi.apply_undistort(
        cam0,
        roi_maps["cam0_map1"],
        roi_maps["cam0_map2"],
    )

    corrected0 = roi.rotate_with_config(
        corrected0,
        roi_distortion_config["CAM0_ROTATION"],
    )

    corrected1 = roi.apply_undistort(
        cam1,
        roi_maps["cam1_map1"],
        roi_maps["cam1_map2"],
    )

    corrected1 = roi.rotate_with_config(
        corrected1,
        roi_distortion_config["CAM1_ROTATION"],
    )

    results = roi.extract_rois(
        corrected0,
        corrected1,
        roi_config,
    )

    if results is None:
        raise RuntimeError("ROI 추출 실패")

    if len(results) != 14:
        raise RuntimeError(
            f"ROI 개수 오류: {len(results)} / 14"
        )

    for index, image in enumerate(results, start=1):
        save_path = (
            REGISTER_ROI_DIR
            / f"original_{index}.png"
        )

        success = roi.imwrite_korean(
            str(save_path),
            image,
            ".png",
        )

        if not success:
            raise RuntimeError(
                f"ROI 저장 실패: {save_path}"
            )

    print(
        f"[ROI] 14개 저장 완료: "
        f"{REGISTER_ROI_DIR}"
    )

    return REGISTER_ROI_DIR


def enter_registration_mode():
    global current_mode, current_id

    current_mode = "registration_id"

    lcd_show(
        "Registration",
        "Input ID"
    )

    time.sleep(0.3)

    user_id = read_id_from_keyboard()

    if user_id is None:
        show_waiting()
        return

    current_id = user_id
    current_mode = "registration_ready"

    lcd_show(
        "Ready to",
        "registrate"
    )

    print(f"[REGISTER] ID={current_id}")
    print("[REGISTER] DOWN key -> capture")


def run_registration():
    global current_mode

    current_mode = "registering"

    lcd_show(
        "Doing",
        "registration"
    )

    try:
        capture_folder = capture_pair()

        if capture_folder is None:
            raise RuntimeError("촬영 실패")

        print(
            f"[REGISTER] capture folder: "
            f"{capture_folder}"
        )

        roi_folder = make_registration_rois(
            capture_folder
        )

        result = enroll_user(
            backbone,
            device,
            current_id,
            str(roi_folder),
        )

        if not result:
            raise RuntimeError("DB 등록 실패")

        lcd_show(
            "Complete",
            ""
        )

        time.sleep(COMPLETE_SECONDS)

    except Exception as e:
        print("[REGISTER ERROR]", repr(e))

        lcd_show(
            "Register error",
            ""
        )

        time.sleep(ERROR_SECONDS)

    finally:
        show_waiting()


def enter_verification_mode():
    global current_mode, current_id

    current_mode = "verification_id"

    lcd_show(
        "Verification",
        "Input ID"
    )

    time.sleep(0.3)

    user_id = read_id_from_keyboard()

    if user_id is None:
        show_waiting()
        return

    if not is_registered_id(user_id):
        lcd_show(
            "Not registered",
            ""
        )

        time.sleep(NOT_REGISTERED_SECONDS)
        show_waiting()
        return

    current_id = user_id
    current_mode = "verification_ready"

    lcd_show(
        "Yes registered",
        "DOWN = capture"
    )

    print(f"[VERIFY] ID={current_id}")
    print("[VERIFY] DOWN key -> capture")


def run_verification():
    global current_mode

    current_mode = "verifying"

    lcd_show(
        "Verification",
        ""
    )

    try:
        capture_folder = capture_pair()

        if capture_folder is None:
            raise RuntimeError("촬영 실패")

        print(
            f"[VERIFY] capture folder: "
            f"{capture_folder}"
        )

        best_signature_path, best_confidence = (
            extract_best_signature(
                yolo_model,
                capture_folder,
                VERIFY_ROI_PATH,
                yolo_device,
            )
        )

        if best_signature_path is None:
            raise RuntimeError(
                "YOLO 서명 검출 실패"
            )

        print(
            f"[YOLO] best confidence="
            f"{best_confidence:.4f}"
        )

        is_genuine = verify_user(
            backbone,
            device,
            current_id,
            str(best_signature_path),
        )

        if is_genuine:
            lcd_show(
                "Verify complete",
                "ORIGINAL"
            )
        else:
            lcd_show(
                "Verify complete",
                "FORGERY"
            )

        time.sleep(RESULT_SECONDS)

    except Exception as e:
        print("[VERIFY ERROR]", repr(e))

        lcd_show(
            "Verify error",
            ""
        )

        time.sleep(ERROR_SECONDS)

    finally:
        show_waiting()

def initialize_system():
    global keyboard, device, backbone
    global yolo_model, yolo_device

    print()
    print("============================================")
    print("SIGNATURE SYSTEM START")
    print("============================================")
    print("LEFT  : Registration")
    print("UP    : Delete user")
    print("DOWN  : Capture")
    print("RIGHT : Verification")
    print("ESC   : Cancel")
    print("============================================")

    init_lcd()

    lcd_show(
        "Starting...",
        ""
    )

    keyboard = find_usb_keyboard()

    if keyboard is None:
        raise RuntimeError(
            "USB 키보드를 찾지 못했습니다."
        )

    device = get_device()
    backbone = load_model(device)

    yolo_model, yolo_device = load_yolo_model()

    init_registration_roi()
    init_cameras()

    show_waiting()

    print("[SYSTEM] Ready.")

def main_loop():
    while True:
        code = read_keypress()

        if current_mode == "waiting":

            if code == ecodes.KEY_LEFT:
                print("[KEY] LEFT -> Registration")
                enter_registration_mode()

            elif code == ecodes.KEY_UP:
                print("[KEY] UP -> Delete user")
                enter_delete_mode()

            elif code == ecodes.KEY_RIGHT:
                print("[KEY] RIGHT -> Verification")
                enter_verification_mode()

        elif current_mode == "registration_ready":

            if code == ecodes.KEY_DOWN:
                print("[KEY] DOWN -> Register capture")
                run_registration()

            elif code == ecodes.KEY_ESC:
                show_waiting()

        elif current_mode == "verification_ready":

            if code == ecodes.KEY_DOWN:
                print("[KEY] DOWN -> Verify capture")
                run_verification()

            elif code == ecodes.KEY_ESC:
                show_waiting()

def cleanup():

    print()
    print("[SYSTEM] Cleaning up...")

    try:
        if lcd is not None:
            lcd.cursor_mode = "hide"
            lcd.clear()

    except Exception:
        pass

    try:
        capture.running = False
        capture.camera1.stop()
        capture.camera2.stop()
        capture.cleanup_light_gpio()

    except Exception:
        pass

if __name__ == "__main__":
    try:
        initialize_system()
        main_loop()

    except KeyboardInterrupt:
        print()
        print("Ctrl+C")

    except Exception as e:
        print("[FATAL]", repr(e))

        try:
            lcd_show(
                "System error",
                ""
            )
        except Exception:
            pass

    finally:
        cleanup()
