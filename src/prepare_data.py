from pathlib import Path
import hashlib

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "flower_photos"
OUTPUT_DIR = PROJECT_DIR / "outputs"
SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    issues = []
    seen = {}

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            relative_path = path.relative_to(PROJECT_DIR).as_posix()

            # 检查图片结构，并实际解码像素
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.convert("RGB").load()
            except Exception as error:
                issues.append({
                    "path": relative_path,
                    "reason": f"invalid image: {error}",
                })
                continue

            # 检查文件内容完全相同的图片
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                previous_path, previous_class = seen[digest]
                if previous_class != class_dir.name:
                    raise ValueError(
                        f"相同图片存在不同标签：{previous_path} 和 {relative_path}"
                    )
                issues.append({
                    "path": relative_path,
                    "reason": f"duplicate of {previous_path}",
                })
                continue

            seen[digest] = (relative_path, class_dir.name)
            records.append({
                "path": relative_path,
                "class": class_dir.name,
                "sha256": digest,
            })

    if not records:
        raise ValueError("没有可用图片，请检查数据路径。")

    data = pd.DataFrame(records)

    # 先留出 30%，再平分为验证集和测试集
    train, remaining = train_test_split(
        data,
        test_size=0.30,
        random_state=SEED,
        stratify=data["class"],
    )
    val, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining["class"],
    )

    parts = []
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame = frame.copy()
        frame["split"] = name
        parts.append(frame)

    manifest = pd.concat(parts, ignore_index=True)

    # 确认同一个文件内容没有跨集合出现
    assert manifest.groupby("sha256")["split"].nunique().max() == 1

    manifest.to_csv(
        OUTPUT_DIR / "dataset_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(issues, columns=["path", "reason"]).to_csv(
        OUTPUT_DIR / "data_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"可用图片：{len(manifest)}")
    print(f"排除的损坏或重复文件：{len(issues)}")
    print("\n各类别划分数量：")
    print(pd.crosstab(manifest["class"], manifest["split"]))
    print("\n各集合总数：")
    print(manifest["split"].value_counts())
    print("\n清单已保存到 outputs/dataset_split.csv")


if __name__ == "__main__":
    main()