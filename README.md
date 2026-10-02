# Dogs vs Cats PyTorch 分类项目

这是一个使用 PyTorch 完成的 Kaggle Dogs vs Cats 猫狗二分类项目。项目包含训练、验证、最佳模型保存、Kaggle 测试集批量预测、`submission.csv` 生成，以及单张图片预测。

## 项目功能

- 从 Kaggle 图片文件名读取类别：`cat.*.jpg` 为猫（0），`dog.*.jpg` 为狗（1）
- 将图片缩放到 128 × 128 并转换为 Tensor
- 使用三组卷积、ReLU 和最大池化构成的 CNN 进行二分类训练
- 按 8:2 划分训练集和验证集，并使用固定随机种子 42
- 记录训练/验证 loss 与 accuracy 到 TensorBoard
- 按验证集准确率保存最佳模型权重
- 加载模型，对 Kaggle 测试集批量预测并通过 Softmax 得到狗类别概率
- 生成 Kaggle 格式的 `outputs/submission.csv`
- 加载 JPG/PNG 单张图片，输出猫、狗概率和最终类别

## 项目目录

```text
dogs vs cats/
├── checkpoints/
│   └── best_model.pth          # 已训练的最佳模型权重（约 8.1 MB）
├── data/
│   ├── predict/
│   │   ├── cat.png             # 单张预测示例：猫
│   │   └── dog.png             # 单张预测示例：狗
│   └── raw/
│       ├── sample_submission.csv
│       ├── train/               # Kaggle 训练图片，本地存在但不提交 Git
│       └── test/                # Kaggle 测试图片，本地存在但不提交 Git
├── logs/                        # TensorBoard 训练日志，不提交 Git
├── outputs/
│   └── submission.csv           # 已生成的 Kaggle 提交文件（12,500 条预测）
├── dataset.py                   # 训练集与 Kaggle 测试集 Dataset
├── model.py                     # CatsDogsCNN 网络定义
├── train.py                     # 训练、验证、日志记录与最佳模型保存
├── submission.py                # 测试集预测与 submission 生成
├── predict.py                   # 单张图片预测
├── requirements.txt             # Python 第三方依赖
├── .gitignore                   # Git 忽略规则
└── README.md
```

## 环境依赖

本项目已在 Python 3.12 环境中进行语法检查。第三方依赖来自源码中的实际 import：

- PyTorch
- torchvision
- pandas
- Pillow
- tensorboard

安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖文件未锁定版本，因为原项目没有保存可核实的包版本信息。安装 PyTorch 时如需匹配特定 CUDA 版本，请按 PyTorch 官方安装方式选择对应构建。

## 数据集

项目使用 Kaggle Dogs vs Cats 数据集。大型原始数据集不包含在 Git 仓库中，请自行下载并整理为：

```text
data/raw/
├── train/
│   ├── cat.0.jpg
│   ├── dog.0.jpg
│   └── ...
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

训练代码根据训练图片文件名的 `cat` / `dog` 前缀生成标签；测试图片文件名应为可转换为整数的编号。当前本地数据规模为 25,000 张训练图片和 12,500 张测试图片。

## 训练方法

在任意工作目录中均可通过脚本路径启动；默认数据位置相对于项目目录解析：

```bash
python train.py
```

训练配置与原项目保持一致：batch size 为 32，训练 10 个 epoch，损失函数为交叉熵，优化器为 Adam，学习率为 0.001。最佳权重保存到 `checkpoints/best_model.pth`。

查看训练日志：

```bash
tensorboard --logdir logs
```

## 测试集预测与 Kaggle submission

准备好 `data/raw/test/` 并确认 `checkpoints/best_model.pth` 存在后运行：

```bash
python submission.py
```

脚本按图片编号排序，将模型输出经 Softmax 转为狗类别概率，结果写入：

```text
outputs/submission.csv
```

仓库保留了一份已生成的 12,500 行提交结果。项目中没有可核实的 Kaggle 分数，因此不声明比赛成绩。

## 单张图片预测

直接运行时默认预测 `data/predict/dog.png`：

```bash
python predict.py
```

也可以传入自己的 JPG 或 PNG 图片：

```bash
python predict.py path/to/your_image.jpg
```

如需使用其他权重：

```bash
python predict.py path/to/your_image.png --model path/to/model.pth
```

输出包含猫概率、狗概率和最终预测类别。

## 模型说明

`CatsDogsCNN` 接收 3 × 128 × 128 的 RGB 图片。网络依次使用 16、32、64 个 3 × 3 卷积核，每层卷积后接 ReLU 和 2 × 2 最大池化；随后展平，通过 128 单元全连接层，最后输出猫、狗两个 logits。项目使用现有结构完成演示，不对模型性能作额外推断。

## 已有结果

- `checkpoints/best_model.pth`：训练过程保存的最佳验证权重
- `outputs/submission.csv`：12,500 条 Kaggle 测试集狗类别概率
- `data/predict/cat.png`、`data/predict/dog.png`：单张预测示例图片

训练日志包含 loss 与 accuracy 标量，但项目未保存可直接引用的汇总报告，因此 README 不虚构具体准确率。

