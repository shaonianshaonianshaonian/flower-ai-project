from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from model import SmallResNet


PROJECT_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_DIR / "outputs" / "run_20261026_183032"

CHINESE_NAMES = {
    "daisy": "雏菊",
    "dandelion": "蒲公英",
    "roses": "玫瑰",
    "sunflowers": "向日葵",
    "tulips": "郁金香",
}


def main():
    # 弹出文件选择窗口
    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="选择一张花卉图片",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not selected:
        print("未选择图片，程序结束。")
        return

    image_path = Path(selected)

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except OSError as error:
        print(f"无法读取图片：{error}")
        return

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
    size = checkpoint["image_size"]

    model = SmallResNet(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 必须与验证、测试时的预处理保持一致
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    inputs = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(inputs)
        scores = torch.softmax(logits, dim=1)[0].cpu()

    ranking = torch.argsort(scores, descending=True).tolist()
    best_index = ranking[0]
    best_class = classes[best_index]

    print(f"图片：{image_path}")
    print(f"预测类别：{CHINESE_NAMES[best_class]} ({best_class})")
    print(f"模型预测分数：{scores[best_index].item():.2%}")

    print("\n各类别分数：")
    for index in ranking:
        name = classes[index]
        print(
            f"{CHINESE_NAMES[name]} ({name})："
            f"{scores[index].item():.2%}"
        )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(image)
    ax.set_title(
        f"Prediction: {best_class}\n"
        f"Softmax score: {scores[best_index].item():.1%}"
    )
    ax.axis("off")
    fig.tight_layout()

    # 不覆盖之前的预测结果
    prediction_dir = RUN_DIR / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    output_path = prediction_dir / f"{image_path.stem}_prediction.png"
    number = 1
    while output_path.exists():
        output_path = prediction_dir / (
            f"{image_path.stem}_prediction_{number}.png"
        )
        number += 1

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\n预测结果图片已保存：{output_path}")


if __name__ == "__main__":
    main()