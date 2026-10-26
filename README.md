# 基于手写小型 ResNet 的五类花卉分类

## 项目简介

使用 PyTorch 手动实现残差块和小型 ResNet，对雏菊、蒲公英、
玫瑰、向日葵和郁金香进行分类。

项目包括数据统计、图片检查、重复数据处理、分层划分、
模型训练、测试集评估和单张图片预测。

模型从零训练，未使用预训练权重；网络为自定义小型 ResNet，
不是标准 ResNet-18。

## 环境

- Windows
- Python 3.11
- PyTorch 2.7.1+cu128
- torchvision 0.22.1+cu128
- 训练显卡：NVIDIA GeForce RTX 5060 Laptop GPU，8 GB 显存

安装命令：

```bash
conda create -n flower-ai python=3.11 pip -y --override-channels -c https://repo.anaconda.com/pkgs/main
conda activate flower-ai
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

以上 GPU 安装方案对应本项目的训练环境，其他硬件需选择适合的 PyTorch 构建。
程序也支持 CPU，但运行速度会不同。

## 数据来源

数据集说明：
https://www.tensorflow.org/tutorials/load_data/images

下载地址：
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

下载后，将压缩包放入 data 文件夹，在项目根目录解压：

```bash
tar -xzf data/flower_photos.tgz -C data
```

解压后的目录：

```text
data/flower_photos/
├── daisy/
├── dandelion/
├── roses/
├── sunflowers/
├── tulips/
└── LICENSE.txt
```

图片许可和作者信息见数据集内的 LICENSE.txt，请保留该文件。

## 数据处理

原始数据共有 3,670 张图片。

程序检查图片是否可读取，并通过 SHA-256 检查文件内容完全相同的重复图片：

- 同类重复图片只保留一份。
- 相同图片存在不同标签时，排除整个冲突组。
- 原始图片不删除，排除原因记录到 outputs/data_issues.csv。
- 当前检查不能识别裁剪、重新压缩等产生的近似重复图片。

本次发现 1 个标签冲突组，总共排除 4 个文件，保留 3,666 张：

| 集合 | 数量 |
|---|---:|
| 训练集 | 2,566 |
| 验证集 | 550 |
| 测试集 | 550 |

按类别分层划分，比例约为 70% / 15% / 15%，随机种子为 42。

## 模型与训练

- 输入：128×128 RGB 图片
- 主干：初始卷积及 4 个残差块
- 通道数：32、64、128、256
- 尺寸变化时使用 1×1 卷积调整捷径分支
- 自适应平均池化后，通过全连接层输出 5 类分数
- 模型参数量：1,227,685
- 损失函数：交叉熵
- 优化器：Adam
- 学习率：0.001
- 权重衰减：0.0001
- 批大小：16
- 正式训练轮数：30
- 训练增强：随机裁剪缩放、随机水平翻转
- 按验证准确率选择最佳模型

## 运行顺序

在项目根目录、激活 flower-ai 环境后执行：

```bash
python check_env.py
python src/analyze_data.py
python src/prepare_data.py
python src/model.py
python src/train.py
```

每次训练会生成独立的 outputs/run_日期_时间 目录。

训练完成后，先将 src/evaluate.py 和 src/predict.py 中的 RUN_DIR
修改为需要使用的训练结果目录，再运行：

```bash
python src/evaluate.py
python src/predict.py
```

预测程序会弹出图片选择窗口，输出各类别分数并保存结果图片。

## 实验结果

本次正式实验目录：outputs/run_20261026_183032

- 最佳模型：第 28 轮
- 最佳验证准确率：78.00%
- 测试准确率：75.82%
- 测试集宏平均 F1：0.7503
- 测试集加权平均 F1：0.7533

| 类别 | 精确率 | 召回率 | F1 |
|---|---:|---:|---:|
| daisy | 0.7595 | 0.6316 | 0.6897 |
| dandelion | 0.6906 | 0.9259 | 0.7911 |
| roses | 0.6989 | 0.6771 | 0.6878 |
| sunflowers | 0.8636 | 0.9048 | 0.8837 |
| tulips | 0.8276 | 0.6050 | 0.6990 |

向日葵的 F1 最高。蒲公英召回率较高，但存在其他类别被误判为
蒲公英的情况。雏菊和郁金香仍有较多漏判，具体错误见混淆矩阵。

固定随机种子有助于复现，但不同硬件和计算环境下结果仍可能有差异。

## 结果文件

训练目录中包括：

- best_model.pth：最佳模型权重、类别映射和训练配置
- history.csv：训练记录
- training_curves.png：训练曲线
- dataset_split.csv：本次训练使用的数据清单
- classification_report.txt：分类报告
- test_metrics.json：评估指标
- test_predictions.csv：逐张测试图片的预测
- confusion_matrix.png：混淆矩阵
- predictions/：单张图片预测结果

数据和训练输出默认不纳入 Git，需按说明重新生成，
或从项目交付包获得对应结果。

## 使用限制

模型只学习了五类花卉。输入其他物体时，仍会输出这五类中的一个，
当前不具备未知类别识别能力。

Softmax 分数表示模型在五类之间的相对分数，
不代表预测正确的真实概率。