# Task 5：基于 KMNIST 的手写字符识别

这是一个单字符 OCR 课程实验。输入是一张 28×28 的灰度字符图，程序输出它所属的日文平假名类别。项目比较了一个全连接基线模型和一个改进 CNN，训练、测试和报告中的成绩都来自本机实际运行。

## 数据集

实验使用 [KMNIST](https://codh.rois.ac.jp/kmnist/)，包含 60,000 张训练图和 10,000 张测试图，共 10 类。代码通过 `torchvision.datasets.KMNIST` 自动下载数据，原始数据采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可。

类别顺序如下：

| 标签 | 字符 | 罗马音 | 标签 | 字符 | 罗马音 |
|---:|:---:|:---:|---:|:---:|:---:|
| 0 | お | o | 5 | は | ha |
| 1 | き | ki | 6 | ま | ma |
| 2 | す | su | 7 | や | ya |
| 3 | つ | tsu | 8 | れ | re |
| 4 | な | na | 9 | を | wo |

## 环境安装

在项目根目录执行：

```bash
python -m pip install -r task5/requirements.txt
```

代码会自动选择 CUDA；没有显卡时会回退到 CPU。

## 复现实验

```bash
# 训练全连接基线模型
python task5/train.py --model baseline --epochs 10 --seed 2026

# 训练改进 CNN
python task5/train.py --model improved --epochs 10 --seed 2026

# 汇总两组结果并生成柱状图
python task5/compare_results.py
```

第一次训练会把数据下载到 `task5/data/`。每次训练都会保存 `metrics.json`、`history.json`、训练曲线、混淆矩阵和预测样例。模型权重 `best.pt` 和原始数据已写入 `.gitignore`，需要时可按上面的命令重新生成。

## 独立评估与单图预测

```bash
# 在官方测试集上重新评估
python task5/evaluate.py \
  --checkpoint task5/results/improved/best.pt

# 识别自己的单字符图片，返回概率最高的 3 类
python task5/predict.py \
  --checkpoint task5/results/improved/best.pt \
  --image /path/to/character.png --top-k 3
```

单图预测会自动转成灰度图。若图片是白底黑字，程序会先反色；普通外部图片会裁剪字符区域，再缩放并居中到 28×28。已经是 28×28 的图片会被视为对齐完成，直接采用与训练集相同的归一化流程，避免重复缩放。模型只学习了 KMNIST 的 10 个类别，因此不能用来识别类别表以外的汉字、英文或数字。

## 本次运行结果

| 模型 | 参数量 | 测试准确率 | Macro-F1 | 训练时间 |
|---|---:|---:|---:|---:|
| 基线 MLP | 101,770 | 88.92% | 88.92% | 12.60 秒 |
| 改进 CNN | 468,010 | 96.72% | 96.71% | 17.35 秒 |

运行环境为 RTX 4090、PyTorch 2.14.0、CUDA 13.0。不同电脑的训练时间会有差异，详细配置与结果见 [实验报告](实验报告.md) 和 `results/*/metrics.json`。

## 测试

```bash
pytest -q task5/tests
ruff check task5
```

当前共有 9 个单元测试，覆盖模型输出形状、类别映射、分层数据划分、指标口径和外部图片预处理。

## 主要文件

```text
task5/
├── data_utils.py          # 数据加载、增强、分层划分和图片预处理
├── models.py              # 基线 MLP 与改进 CNN
├── experiment_utils.py    # 训练、评估、指标和作图工具
├── train.py               # 训练并在测试集上评估
├── evaluate.py            # 独立测试集评估
├── predict.py             # 单张外部图片识别
├── compare_results.py     # 两组实验汇总与对比图
├── results/               # 指标、曲线、混淆矩阵和预测样例
├── tests/                 # 不依赖联网的单元测试
└── 实验报告.md             # 中文实验报告
```
