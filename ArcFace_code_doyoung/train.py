import os
import csv
import json
import time
import random
from datetime import datetime

import torch
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    Subset
)

from torchvision import (
    datasets,
    transforms
)

from torchvision.transforms import (
    functional as TF,
    InterpolationMode
)

from model import (
    SignatureBackbone,
    ArcFaceLoss
)

class ResizeWithPadding:

    def __init__(
        self,
        target_width=320,
        target_height=224
    ):

        self.target_width = (
            target_width
        )

        self.target_height = (
            target_height
        )


    def __call__(
        self,
        image
    ):

        image = image.convert(
            "RGB"
        )


        width, height = (
            image.size
        )


        if width <= 0 or height <= 0:

            raise ValueError(
                "잘못된 이미지 크기"
            )

        scale = min(

            self.target_width
            /
            width,

            self.target_height
            /
            height
        )


        new_width = max(
            1,
            int(
                round(
                    width
                    *
                    scale
                )
            )
        )


        new_height = max(
            1,
            int(
                round(
                    height
                    *
                    scale
                )
            )
        )

        image = TF.resize(

            image,

            [
                new_height,
                new_width
            ],

            interpolation=(
                InterpolationMode.BILINEAR
            ),

            antialias=True
        )

        pad_left = (

            self.target_width
            -
            new_width

        ) // 2


        pad_right = (

            self.target_width
            -
            new_width
            -
            pad_left
        )


        pad_top = (

            self.target_height
            -
            new_height

        ) // 2


        pad_bottom = (

            self.target_height
            -
            new_height
            -
            pad_top
        )

        image = TF.pad(

            image,

            [
                pad_left,
                pad_top,
                pad_right,
                pad_bottom
            ],

            fill=255
        )


        return image

def create_stratified_split(
    targets,
    val_ratio=0.2,
    seed=42
):

    class_indices = {}


    for index, label in enumerate(
        targets
    ):

        if label not in class_indices:

            class_indices[
                label
            ] = []


        class_indices[
            label
        ].append(
            index
        )


    rng = random.Random(
        seed
    )


    train_indices = []
    val_indices = []

    for label, indices in (
        class_indices.items()
    ):

        indices = (
            indices.copy()
        )


        rng.shuffle(
            indices
        )


        num_images = len(
            indices
        )

        if num_images == 1:

            train_indices.extend(
                indices
            )

            continue

        num_val = max(
            1,
            int(
                round(
                    num_images
                    *
                    val_ratio
                )
            )
        )

        num_val = min(
            num_val,
            num_images - 1
        )


        val_indices.extend(
            indices[
                :num_val
            ]
        )


        train_indices.extend(
            indices[
                num_val:
            ]
        )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        val_indices
    )


    return (
        train_indices,
        val_indices
    )

def calculate_predictions(
    embeddings,
    arcface_loss
):

    normalized_weight = F.normalize(
        arcface_loss.weight,
        p=2,
        dim=1
    )


    logits = F.linear(
        embeddings,
        normalized_weight
    )


    predictions = torch.argmax(
        logits,
        dim=1
    )


    return predictions

def train_one_epoch(
    backbone,
    arcface_loss,
    dataloader,
    optimizer,
    scaler,
    device
):

    backbone.train()
    arcface_loss.train()


    total_loss = 0.0
    total_correct = 0
    total_samples = 0


    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )


        labels = labels.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )

        if scaler is not None:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                embeddings = backbone(
                    images
                )


                loss = arcface_loss(
                    embeddings,
                    labels
                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()

        else:

            embeddings = backbone(
                images
            )


            loss = arcface_loss(
                embeddings,
                labels
            )


            loss.backward()


            optimizer.step()

        with torch.no_grad():

            predictions = (
                calculate_predictions(
                    embeddings.detach(),
                    arcface_loss
                )
            )


            correct = (
                predictions
                ==
                labels
            ).sum().item()


        batch_size = (
            images.size(0)
        )


        total_loss += (
            loss.item()
            *
            batch_size
        )


        total_correct += (
            correct
        )


        total_samples += (
            batch_size
        )

    average_loss = (
        total_loss
        /
        total_samples
    )


    accuracy = (
        total_correct
        /
        total_samples
        *
        100.0
    )


    return (
        average_loss,
        accuracy
    )

def validate_one_epoch(
    backbone,
    arcface_loss,
    dataloader,
    device
):

    backbone.eval()
    arcface_loss.eval()


    total_loss = 0.0
    total_correct = 0
    total_samples = 0


    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                device,
                non_blocking=True
            )


            labels = labels.to(
                device,
                non_blocking=True
            )

            if device.type == "cuda":

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    embeddings = backbone(
                        images
                    )


                    loss = arcface_loss(
                        embeddings,
                        labels
                    )

            else:

                embeddings = backbone(
                    images
                )


                loss = arcface_loss(
                    embeddings,
                    labels
                )


            predictions = (
                calculate_predictions(
                    embeddings,
                    arcface_loss
                )
            )


            correct = (
                predictions
                ==
                labels
            ).sum().item()


            batch_size = (
                images.size(0)
            )


            total_loss += (
                loss.item()
                *
                batch_size
            )


            total_correct += (
                correct
            )


            total_samples += (
                batch_size
            )


    if total_samples == 0:

        return (
            float("inf"),
            0.0
        )


    average_loss = (
        total_loss
        /
        total_samples
    )


    accuracy = (
        total_correct
        /
        total_samples
        *
        100.0
    )


    return (
        average_loss,
        accuracy
    )

def initialize_log_file(
    log_path
):

    if os.path.exists(
        log_path
    ):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


        backup_path = (
            os.path.splitext(
                log_path
            )[0]
            +
            f"_backup_{timestamp}.csv"
        )


        os.rename(
            log_path,
            backup_path
        )


        print(
            f"기존 로그 백업: "
            f"{backup_path}"
        )

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
                "epoch_seconds"
            ]
        )

def append_log(
    log_path,
    epoch,
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
    learning_rate,
    epoch_seconds
):

    with open(
        log_path,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                epoch,

                f"{train_loss:.6f}",

                f"{train_accuracy:.4f}",

                f"{val_loss:.6f}",

                f"{val_accuracy:.4f}",

                f"{learning_rate:.10f}",

                f"{epoch_seconds:.2f}"
            ]
        )

def save_best_summary(
    path,
    best_val_accuracy,
    best_accuracy_epoch,
    best_val_loss,
    best_loss_epoch
):

    data = {

        "best_val_accuracy":
            best_val_accuracy,

        "best_accuracy_epoch":
            best_accuracy_epoch,

        "best_val_loss":
            best_val_loss,

        "best_loss_epoch":
            best_loss_epoch
    }


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

def train():

    torch.backends.cudnn.benchmark = True


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(
        f"학습 장치(Device): "
        f"{device}"
    )


    # ========================================================
    #                  하이퍼파라미터
    # ========================================================

    DATA_DIR = "./dataset/train"

    SAVE_DIR = "./checkpoints"

    LOG_DIR = "./logs"


    EMBEDDING_SIZE = 128

    INPUT_WIDTH = 320
    INPUT_HEIGHT = 224


    BATCH_SIZE = 32

    EPOCHS = 300

    LEARNING_RATE = 0.0001

    NUM_WORKERS = 2

    VALIDATION_RATIO = 0.1


    RANDOM_SEED = 42

    if not os.path.exists(
        DATA_DIR
    ):

        print(
            f" 에러: "
            f"'{DATA_DIR}' 폴더가 없습니다."
        )

        return


    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )


    os.makedirs(
        LOG_DIR,
        exist_ok=True
    )

    LOG_PATH = os.path.join(
        LOG_DIR,
        "training_log.csv"
    )


    BEST_SUMMARY_PATH = os.path.join(
        LOG_DIR,
        "best_epochs.json"
    )


    initialize_log_file(
        LOG_PATH
    )

    data_transform = transforms.Compose(
        [

            ResizeWithPadding(
                target_width=INPUT_WIDTH,
                target_height=INPUT_HEIGHT
            ),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ]
    )

    full_dataset = datasets.ImageFolder(

        root=DATA_DIR,

        transform=data_transform
    )


    NUM_CLASSES = len(
        full_dataset.classes
    )


    if NUM_CLASSES < 2:

        print(
            " 작성자가 2명 미만입니다."
        )

        return

    train_indices, val_indices = (
        create_stratified_split(

            full_dataset.targets,

            val_ratio=(
                VALIDATION_RATIO
            ),

            seed=(
                RANDOM_SEED
            )
        )
    )


    if len(val_indices) == 0:

        print()
        print(
            " Validation 데이터가 없습니다."
        )

        print(
            "작성자별 이미지 수를 확인하세요."
        )

        return


    train_dataset = Subset(
        full_dataset,
        train_indices
    )


    val_dataset = Subset(
        full_dataset,
        val_indices
    )

    train_drop_last = (

        len(train_dataset)
        %
        BATCH_SIZE
        ==
        1

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type
            ==
            "cuda"
        ),

        drop_last=train_drop_last,

        persistent_workers=(
            NUM_WORKERS
            >
            0
        )
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            device.type
            ==
            "cuda"
        ),

        drop_last=False,

        persistent_workers=(
            NUM_WORKERS
            >
            0
        )
    )

    print()
    print(
        "============================================"
    )

    print(
        "데이터셋 로드 완료"
    )

    print(
        "============================================"
    )


    print(
        f"전체 이미지    : "
        f"{len(full_dataset)}장"
    )


    print(
        f"Train 이미지   : "
        f"{len(train_dataset)}장"
    )


    print(
        f"Validation 이미지: "
        f"{len(val_dataset)}장"
    )


    print(
        f"작성자 수       : "
        f"{NUM_CLASSES}명"
    )


    print(
        f"입력 크기       : "
        f"{INPUT_WIDTH} x "
        f"{INPUT_HEIGHT}"
    )


    print(
        f"Batch Size      : "
        f"{BATCH_SIZE}"
    )


    print(
        f"Epochs          : "
        f"{EPOCHS}"
    )


    print(
        "============================================"
    )

    backbone = SignatureBackbone(

        embedding_size=(
            EMBEDDING_SIZE
        )

    ).to(
        device
    )

    arcface_loss = ArcFaceLoss(

        num_classes=(
            NUM_CLASSES
        ),

        embedding_size=(
            EMBEDDING_SIZE
        ),

        s=30.0,

        m=0.5

    ).to(
        device
    )

    optimizer = torch.optim.Adam(

        list(
            backbone.parameters()
        )

        +

        list(
            arcface_loss.parameters()
        ),

        lr=LEARNING_RATE
    )

    scaler = (

        torch.amp.GradScaler(
            "cuda"
        )

        if device.type
        ==
        "cuda"

        else None
    )

    best_val_accuracy = (
        -1.0
    )

    best_accuracy_epoch = (
        0
    )


    best_val_loss = (
        float("inf")
    )

    best_loss_epoch = (
        0
    )


    last_completed_epoch = (
        0
    )

    print()
    print(
        "--- 학습 시작 "
        "(중단: Ctrl+C 또는 stop.txt 생성) ---"
    )

    print()


    try:

        for epoch in range(
            1,
            EPOCHS + 1
        ):

            if os.path.exists(
                "stop.txt"
            ):

                print()
                print(
                    "'stop.txt' 파일 감지!"
                )

                print(
                    f"Epoch "
                    f"{last_completed_epoch}까지 "
                    f"완료 후 안전 종료합니다."
                )


                os.remove(
                    "stop.txt"
                )

                break


            epoch_start_time = (
                time.time()
            )

            train_loss, train_accuracy = (
                train_one_epoch(

                    backbone,

                    arcface_loss,

                    train_loader,

                    optimizer,

                    scaler,

                    device
                )
            )

            val_loss, val_accuracy = (
                validate_one_epoch(

                    backbone,

                    arcface_loss,

                    val_loader,

                    device
                )
            )


            epoch_elapsed_time = (

                time.time()
                -
                epoch_start_time

            )


            learning_rate = (
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )


            last_completed_epoch = (
                epoch
            )

            print(
                f"Epoch "
                f"[{epoch}/{EPOCHS}]"
            )


            print(
                f"  Train Loss : "
                f"{train_loss:.4f}"
            )


            print(
                f"  Train Acc  : "
                f"{train_accuracy:.2f}%"
            )


            print(
                f"  Val Loss   : "
                f"{val_loss:.4f}"
            )


            print(
                f"  Val Acc    : "
                f"{val_accuracy:.2f}%"
            )


            print(
                f"  LR         : "
                f"{learning_rate:.8f}"
            )


            print(
                f"  Time       : "
                f"{epoch_elapsed_time:.2f} sec"
            )

            epoch_path = os.path.join(

                SAVE_DIR,

                f"signature_backbone_epoch_"
                f"{epoch}.pth"
            )


            torch.save(

                backbone.state_dict(),

                epoch_path
            )

            latest_path = os.path.join(

                SAVE_DIR,

                "signature_backbone_latest.pth"
            )


            torch.save(

                backbone.state_dict(),

                latest_path
            )

            torch.save(

                backbone.state_dict(),

                "signature_backbone.pth"
            )


            print(
                f"  💾 저장: "
                f"{epoch_path}"
            )

            if (
                val_accuracy
                >
                best_val_accuracy
            ):

                best_val_accuracy = (
                    val_accuracy
                )


                best_accuracy_epoch = (
                    epoch
                )


                best_accuracy_path = (
                    os.path.join(

                        SAVE_DIR,

                        "best_val_accuracy.pth"
                    )
                )


                torch.save(

                    backbone.state_dict(),

                    best_accuracy_path
                )


                print(
                    f"   Best Val Accuracy 갱신: "
                    f"{val_accuracy:.2f}% "
                    f"(Epoch {epoch})"
                )

            if (
                val_loss
                <
                best_val_loss
            ):

                best_val_loss = (
                    val_loss
                )


                best_loss_epoch = (
                    epoch
                )


                best_loss_path = (
                    os.path.join(

                        SAVE_DIR,

                        "best_val_loss.pth"
                    )
                )


                torch.save(

                    backbone.state_dict(),

                    best_loss_path
                )


                print(
                    f"   Best Val Loss 갱신: "
                    f"{val_loss:.4f} "
                    f"(Epoch {epoch})"
                )

            append_log(

                LOG_PATH,

                epoch,

                train_loss,

                train_accuracy,

                val_loss,

                val_accuracy,

                learning_rate,

                epoch_elapsed_time
            )

            save_best_summary(

                BEST_SUMMARY_PATH,

                best_val_accuracy,

                best_accuracy_epoch,

                best_val_loss,

                best_loss_epoch
            )


            print()

    except KeyboardInterrupt:

        print()
        print(
            "사용자에 의해 학습이 "
            "중단되었습니다."
        )

    print()
    print(
        "============================================"
    )

    print(
        "학습 종료"
    )

    print(
        "============================================"
    )


    if last_completed_epoch > 0:

        print(
            f"마지막 완료 Epoch : "
            f"{last_completed_epoch}"
        )


        print(
            f"Best Val Accuracy : "
            f"{best_val_accuracy:.2f}% "
            f"(Epoch "
            f"{best_accuracy_epoch})"
        )


        print(
            f"Best Val Loss     : "
            f"{best_val_loss:.4f} "
            f"(Epoch "
            f"{best_loss_epoch})"
        )


        print()
        print(
            f"Checkpoint 폴더:"
        )

        print(
            SAVE_DIR
        )


        print()
        print(
            f"학습 로그:"
        )

        print(
            LOG_PATH
        )


        print()
        print(
            f"Best Epoch 정보:"
        )

        print(
            BEST_SUMMARY_PATH
        )


    else:

        print(
            "완료된 Epoch가 없습니다."
        )


    print(
        "============================================"
    )


if __name__ == "__main__":

    train()