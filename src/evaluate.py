from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from model import SmallResNet
from train import FlowerDataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_DIR / "outputs" / "run_20261026_183032"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 仅加载自己训练生成的模型文件
    checkpoint = torch.load(
        RUN_DIR / "best_model.pth",
        map_location="cpu",
        weights_only=True,
    )

    class_to_idx = checkpoint["class_to_idx"]
    classes = sorted(class_to_idx, key=class_to_idx.get)
    image_size = checkpoint["image_size"]

    model = SmallResNet(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 使用该次训练保存的清单，确保评估对应同一次实验
    manifest = pd.read_csv(RUN_DIR / "dataset_split.csv")
    test_frame = manifest[
        manifest["split"] == "test"
    ].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = FlowerDataset(test_frame, class_to_idx, transform)
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0
    )

    targets = []
    predictions = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())

    accuracy = accuracy_score(targets, predictions)
    label_ids = list(range(len(classes)))

    report = classification_report(
        targets,
        predictions,
        labels=label_ids,
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    print(f"加载模型来自第 {checkpoint['epoch']} 轮")
    print(f"最佳验证准确率：{checkpoint['val_acc']:.2%}")
    print(f"测试图片数量：{len(targets)}")
    print(f"测试准确率：{accuracy:.2%}")
    print("\n各类别评估：")
    print(report)

    (RUN_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    metrics = {
        "checkpoint_epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["val_acc"],
        "test_accuracy": accuracy,
        "test_count": len(targets),
    }
    (RUN_DIR / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 保存每张测试图片的预测，便于分析错误案例
    results = test_frame[["path", "class"]].copy()
    results = results.rename(columns={"class": "true_class"})
    results["predicted_class"] = [
        classes[index] for index in predictions
    ]
    results["correct"] = [
        actual == predicted
        for actual, predicted in zip(targets, predictions)
    ]
    results.to_csv(
        RUN_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        targets,
        predictions,
        labels=label_ids,
        display_labels=classes,
        cmap="Blues",
        ax=ax,
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Test Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\n评估结果已保存到：{RUN_DIR}")


if __name__ == "__main__":
    main()