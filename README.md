# code-snippets

可直接运行、用途明确的小型 Python 工具和示例。

## 目录

| 路径 | 用途 | 主要依赖 |
| --- | --- | --- |
| `diagnostics/check_pytorch.py` | 检查 PyTorch、CUDA、cuDNN 和 GPU 张量计算 | PyTorch |
| `diagnostics/collect_pytorch_env.py` | 调用 PyTorch 官方环境收集器 | PyTorch |
| `examples/mnist_train.py` | 可配置的 MNIST 训练示例 | PyTorch、TorchVision |
| `vision/overlay_mask_contours.py` | 将眼底分割掩膜轮廓叠加到原图 | NumPy、OpenCV |
| [VPS_Setup/DMIT_xray_flclash](VPS_Setup/DMIT_xray_flclash/SKILL.md) | DMIT HY2/REALITY、可选 WARP 与多平台分流技能 | Python 标准库；目标代理内核另行安装 |

## 安装

建议先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

PyTorch/CUDA 的安装命令取决于操作系统、GPU 和 CUDA 版本，请使用 [PyTorch 官方安装选择器](https://docs.pytorch.org/get-started/locally/)。图像工具的依赖可直接安装：

```bash
python -m pip install -r requirements/vision.txt
```

## 使用

### 检查 PyTorch 和 CUDA

```bash
python diagnostics/check_pytorch.py
python diagnostics/check_pytorch.py --json
python diagnostics/check_pytorch.py --require-cuda
```

`--require-cuda` 在 CUDA 不可用或 GPU 张量测试失败时返回非零退出码，适合服务器部署后的自动检查。

完整环境报告：

```bash
python diagnostics/collect_pytorch_env.py
```

### 训练 MNIST

```bash
python examples/mnist_train.py --epochs 10
python examples/mnist_train.py --device cpu --epochs 1 --dry-run
```

使用 `--help` 查看批大小、学习率、数据目录和模型保存路径等参数。

### 叠加分割轮廓

```bash
python vision/overlay_mask_contours.py \
  input/image.png \
  input/ground_truth_mask.png \
  outputs/overlay.png \
  --prediction-mask input/prediction_mask.png \
  --size 384 384
```

默认按眼底视盘/视杯标签处理：背景为 `0`，非背景区域为视盘，`255` 为视杯。真值轮廓使用红/黄，预测轮廓使用蓝/绿。可通过参数修改背景标签、视杯标签和线宽。

## 原脚本对应关系

| 原文件 | 整理后 |
| --- | --- |
| `collect_env.py` | `diagnostics/collect_pytorch_env.py` |
| `gpu_test.py`、`cuda_cudnn_install_eval.py` | `diagnostics/check_pytorch.py` |
| `mnist_train.py` | `examples/mnist_train.py` |
| `overlay_mask_contour_on_image.py` | `vision/overlay_mask_contours.py` |

数据、虚拟环境和 `outputs/` 不纳入版本控制。
