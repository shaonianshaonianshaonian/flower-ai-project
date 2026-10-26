from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import SmallResNet


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

SEED = 42
EPOCHS = 30          # 先检查流程，之后再增加
BATCH_SIZE = 16
LEARNING_RATE = 0.001
IMAGE_SIZE = 128


class FlowerDataset(Dataset):
    def __init__(self, frame, class_to_idx, transform):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = PROJECT_DIR / row["path"]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = self.class_to_idx[row["class"]]
        return image, label


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(
            loader, desc="Train" if training else "Val"
        ):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += count

    return total_loss / total, correct / total


def save_curves(history, run_dir):
    frame = pd.DataFrame(history)
    frame.to_csv(run_dir / "history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(frame["epoch"], frame["train_acc"], label="Train")
    axes[1].plot(frame["epoch"], frame["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"训练设备：{device}")

    manifest_path = OUTPUT_DIR / "dataset_split.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("请先运行 prepare_data.py")

    manifest = pd.read_csv(manifest_path)
    classes = sorted(manifest["class"].unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    print(f"类别映射：{class_to_idx}")

    # 每次训练单独保存，避免覆盖上一次实验
    run_dir = OUTPUT_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    # 保存本次使用的数据清单，方便复现
    manifest.to_csv(run_dir / "dataset_split.csv", index=False)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE, scale=(0.7, 1.0)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    train_data = FlowerDataset(
        manifest[manifest["split"] == "train"],
        class_to_idx,
        train_transform,
    )
    val_data = FlowerDataset(
        manifest[manifest["split"] == "val"],
        class_to_idx,
        val_transform,
    )

    # Windows 下先用单进程加载，降低初次运行的复杂度
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    model = SmallResNet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    history = []
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "image_size": IMAGE_SIZE,
                "epoch": epoch,
                "val_acc": val_acc,
                "config": {
                    "seed": SEED,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": 1e-4,
                    "model": "SmallResNet",
                },
            }, run_dir / "best_model.pth")

        save_curves(history, run_dir)

        elapsed = time.perf_counter() - start
        print(
            f"训练损失：{train_loss:.4f} | "
            f"训练准确率：{train_acc:.2%}\n"
            f"验证损失：{val_loss:.4f} | "
            f"验证准确率：{val_acc:.2%}\n"
            f"本轮耗时：{elapsed:.1f} 秒"
        )

    print(f"\n训练结束，最佳验证准确率：{best_val_acc:.2%}")
    print(f"结果目录：{run_dir}")


if __name__ == "__main__":
    main()