from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 根据代码文件的位置定位项目，不受运行目录影响
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "flower_photos"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"数据目录不存在：{DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 每个子文件夹代表一个类别
    records = []
    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        count = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        records.append({"class": class_dir.name, "count": count})

    if not records:
        raise ValueError("没有找到类别文件夹，请检查数据目录。")

    # Pandas：整理统计表
    stats = pd.DataFrame(records)

    # NumPy：计算总数、平均数量和类别数量比
    counts = stats["count"].to_numpy()
    total = int(np.sum(counts))
    if np.min(counts) == 0:
        raise ValueError("存在没有图片的类别，请检查数据。")

    stats["percentage"] = (counts / total * 100).round(2)

    print("各类别图片数量：")
    print(stats.to_string(index=False))
    print(f"\n图片总数：{total}")
    print(f"每类平均数量：{np.mean(counts):.1f}")
    print(f"最多与最少类别的数量比：{np.max(counts) / np.min(counts):.2f}")

    csv_path = OUTPUT_DIR / "class_counts.csv"
    stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Matplotlib：绘制类别分布
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stats["class"], stats["count"], color="steelblue")
    ax.bar_label(bars, padding=3)
    ax.set_title("Flower Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, int(np.max(counts) * 1.15))
    fig.tight_layout()

    chart_path = OUTPUT_DIR / "class_distribution.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(f"\n统计表已保存：{csv_path}")
    print(f"柱状图已保存：{chart_path}")


if __name__ == "__main__":
    main()