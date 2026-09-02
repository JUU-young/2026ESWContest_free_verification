import os
import sqlite3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import SignatureBackbone
from signature_preprocess import preprocess_signature


MODEL_PATH = "./signature_backbone_epoch_266.pth"
DATABASE_PATH = "./database/signature.db"

EMBEDDING_SIZE = 128
MARGIN_TENSION = 0.1

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
)


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_model(device):

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"가중치 파일이 없습니다: {MODEL_PATH}"
        )

    backbone = SignatureBackbone(
        embedding_size=EMBEDDING_SIZE,
        pretrained=False
    ).to(device)

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device
        )

    backbone.load_state_dict(state_dict)
    backbone.eval()

    print(f"Device : {device}")
    print(f"Model  : {MODEL_PATH}")

    return backbone


def connect_database():

    database_dir = os.path.dirname(
        DATABASE_PATH
    )

    if database_dir:
        os.makedirs(
            database_dir,
            exist_ok=True
        )

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            centroid BLOB NOT NULL,
            threshold REAL NOT NULL
        )
        """
    )

    conn.commit()

    return conn


def vector_to_blob(vector):

    array = np.asarray(
        vector,
        dtype=np.float32
    )

    if array.shape != (
        EMBEDDING_SIZE,
    ):
        raise ValueError(
            f"centroid 크기 오류: {array.shape}"
        )

    return array.tobytes()


def blob_to_tensor(
    blob,
    device
):

    array = np.frombuffer(
        blob,
        dtype=np.float32
    ).copy()

    if array.shape[0] != EMBEDDING_SIZE:
        raise ValueError(
            f"DB centroid 차원 오류: "
            f"{array.shape[0]} "
            f"(기대값 {EMBEDDING_SIZE})"
        )

    tensor = torch.from_numpy(
        array
    ).to(
        device=device,
        dtype=torch.float32
    )

    return F.normalize(
        tensor,
        p=2,
        dim=0
    )


def get_embedding(
    backbone,
    image_path,
    device
):

    tensor = preprocess_signature(
        str(image_path),
        add_batch_dimension=True
    )

    if tensor is None:
        return None

    tensor = tensor.to(
        device,
        non_blocking=True
    )

    with torch.no_grad():
        embedding = backbone(
            tensor
        )

    embedding = embedding.squeeze(
        0
    )

    return F.normalize(
        embedding,
        p=2,
        dim=0
    )


def get_image_files(folder):

    folder = Path(
        folder
    )

    if not folder.is_dir():
        raise NotADirectoryError(
            f"폴더가 없습니다: {folder}"
        )

    return sorted(
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in VALID_EXTENSIONS
        )
    )


def enroll_user(
    backbone,
    device,
    user_id,
    folder
):

    image_files = get_image_files(
        folder
    )

    if len(image_files) < 2:
        print(
            "[ERROR] 등록 서명은 "
            "최소 2장 이상 필요합니다."
        )
        return False

    print()
    print("============================================")
    print("서명 등록")
    print("============================================")
    print(f"ID     : {user_id}")
    print(f"Folder : {folder}")
    print(f"Images : {len(image_files)}")
    print()

    enroll_vectors = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"[{index}/{len(image_files)}] "
            f"{image_path.name}"
        )

        vector = get_embedding(
            backbone,
            image_path,
            device
        )

        if vector is None:
            print("  -> 처리 실패")
            continue

        enroll_vectors.append(
            vector
        )

        print(
            "  -> 128-D embedding 완료"
        )

    if len(enroll_vectors) < 2:
        print(
            "[ERROR] 정상 처리된 등록 서명이 "
            "2장 미만입니다."
        )
        return False

    stacked = torch.stack(
        enroll_vectors,
        dim=0
    )

    centroid = torch.mean(
        stacked,
        dim=0
    )

    centroid = F.normalize(
        centroid,
        p=2,
        dim=0
    )

    internal_sims = [
        torch.dot(
            centroid,
            vector
        ).item()
        for vector in enroll_vectors
    ]

    minimum_internal_sim = min(
        internal_sims
    )

    mean_internal_sim = float(
        np.mean(
            internal_sims
        )
    )

    threshold = (
        minimum_internal_sim
        -
        MARGIN_TENSION
    )

    centroid_numpy = (
        centroid
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    centroid_blob = vector_to_blob(
        centroid_numpy
    )

    conn = connect_database()

    try:
        conn.execute(
            """
            INSERT INTO users (
                id,
                centroid,
                threshold
            )
            VALUES (?, ?, ?)
            ON CONFLICT(id)
            DO UPDATE SET
                centroid = excluded.centroid,
                threshold = excluded.threshold
            """,
            (
                str(user_id),
                sqlite3.Binary(
                    centroid_blob
                ),
                float(
                    threshold
                )
            )
        )

        conn.commit()

    finally:
        conn.close()

    print()
    print("--------------------------------------------")
    print("등록 완료")
    print("--------------------------------------------")
    print(f"ID              : {user_id}")
    print(f"Centroid        : {EMBEDDING_SIZE}차원")
    print(
        f"평균 내부 유사도 : "
        f"{mean_internal_sim:.6f}"
    )
    print(
        f"최소 내부 유사도 : "
        f"{minimum_internal_sim:.6f}"
    )
    print(
        f"Threshold       : "
        f"{threshold:.6f}"
    )
    print(
        f"Database        : "
        f"{DATABASE_PATH}"
    )
    print("============================================")

    return True


def verify_user(
    backbone,
    device,
    user_id,
    image_path
):

    conn = connect_database()

    try:
        row = conn.execute(
            """
            SELECT
                centroid,
                threshold
            FROM users
            WHERE id = ?
            """,
            (
                str(
                    user_id
                ),
            )
        ).fetchone()

    finally:
        conn.close()

    if row is None:
        print(
            f"[ERROR] 등록되지 않은 ID입니다: "
            f"{user_id}"
        )
        return False

    centroid_blob, threshold = (
        row
    )

    centroid = blob_to_tensor(
        centroid_blob,
        device
    )

    test_vector = get_embedding(
        backbone,
        image_path,
        device
    )

    if test_vector is None:
        print(
            "[ERROR] 검증 이미지 처리 실패"
        )
        return False

    similarity = torch.dot(
        centroid,
        test_vector
    ).item()

    threshold = float(
        threshold
    )

    is_genuine = (
        similarity
        >=
        threshold
    )

    print()
    print("============================================")
    print("서명 검증")
    print("============================================")
    print(f"ID         : {user_id}")
    print(f"Image      : {image_path}")
    print(f"Similarity : {similarity:.6f}")
    print(f"Threshold  : {threshold:.6f}")
    print("--------------------------------------------")

    if is_genuine:
        print("RESULT     : GENUINE")
        print("판정       : 동일인 서명")
    else:
        print("RESULT     : FORGERY")
        print("판정       : 위조 / 타인 서명")

    print("============================================")

    return is_genuine


def list_users():

    conn = connect_database()

    try:
        rows = conn.execute(
            """
            SELECT
                id,
                threshold
            FROM users
            ORDER BY id
            """
        ).fetchall()

    finally:
        conn.close()

    print()
    print("============================================")
    print("등록 사용자")
    print("============================================")

    if not rows:
        print(
            "등록된 사용자가 없습니다."
        )
    else:
        for user_id, threshold in rows:
            print(
                f"ID={user_id}"
                f" | threshold="
                f"{float(threshold):.6f}"
            )

    print("============================================")


def delete_user(
    user_id
):

    conn = connect_database()

    try:
        cursor = conn.execute(
            """
            DELETE FROM users
            WHERE id = ?
            """,
            (
                str(
                    user_id
                ),
            )
        )

        conn.commit()

        deleted = (
            cursor.rowcount
            >
            0
        )

    finally:
        conn.close()

    if deleted:
        print(
            f"ID {user_id} 삭제 완료"
        )
    else:
        print(
            f"[ERROR] 등록되지 않은 ID입니다: "
            f"{user_id}"
        )


def init_database():

    conn = connect_database()
    conn.close()

    print(
        f"Database 생성/확인 완료: "
        f"{DATABASE_PATH}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Signature SQLite "
            "Enrollment / Verification"
        )
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True
    )

    subparsers.add_parser(
        "init"
    )

    enroll_parser = subparsers.add_parser(
        "enroll"
    )

    enroll_parser.add_argument(
        "--id",
        required=True,
        type=str
    )

    enroll_parser.add_argument(
        "--folder",
        required=True,
        type=str
    )

    verify_parser = subparsers.add_parser(
        "verify"
    )

    verify_parser.add_argument(
        "--id",
        required=True,
        type=str
    )

    verify_parser.add_argument(
        "--image",
        required=True,
        type=str
    )

    subparsers.add_parser(
        "list"
    )

    delete_parser = subparsers.add_parser(
        "delete"
    )

    delete_parser.add_argument(
        "--id",
        required=True,
        type=str
    )

    args = parser.parse_args()

    if args.mode == "init":
        init_database()
        return

    if args.mode == "list":
        list_users()
        return

    if args.mode == "delete":
        delete_user(
            args.id
        )
        return

    device = get_device()

    backbone = load_model(
        device
    )

    if args.mode == "enroll":
        enroll_user(
            backbone,
            device,
            args.id,
            args.folder
        )

    elif args.mode == "verify":
        verify_user(
            backbone,
            device,
            args.id,
            args.image
        )


if __name__ == "__main__":
    main()
