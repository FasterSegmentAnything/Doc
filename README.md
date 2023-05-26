目录
- [说明](#说明)
- [实验环境](#实验环境)
  - [硬件：](#硬件)
  - [软件：](#软件)
- [关于](#关于)

# 说明
分析[segment-anything](https://segment-anything.com/)的工作原理、部署流程、使用技巧及其它。

# 实验环境

> 无特殊指定的情况下，本文档默认指代环境如下：

## 硬件：

| 类型 | 简述 |
| :-- | :-- |
| 名称 | `NVIDIA产品`</br>`Jetson Orin Nano Developer Kit(8GB)` |
|  CPU | `6`核`aarch64 (ARM)` `Cortex-A78AE v8.2` |
|  GPU | `Ampere`架构</br>`1024 NVIDIA CUDA核心` + `32 Tensor核心` | 
| SoC | `tegra23x` |
| 内存/显存 | `共享8GB` |
| 存储 | `128GB M.2 SSD`</br>`无SD卡` |
| 功率 | `7W` ~ `15W` |
| `AI`算力 | `40TOPS` |
| 参考价格（RMB）| `3700` |

查看硬件参数:

```shell
jtop

# 进入后按数字`7`可查看详细信息
```

## 软件：

| 类型 | 简述 |
| :-- | :-- |
| 系统 | `Ubuntu 20.04.5 LTS (GNU/Linux 5.10.104-tegra aarch64)` |
| CUDA | `11.4.315`
| cuDNN | `8.6.0.166` |
| TensorRT | `8.5.2.2` |
| OpenCV | `4.5.4`</br>with CUDA: `NO` |
| Python | `3.8.10`  |
| Vulkan | `1.3.204` |
| Jetpack | `5.1.1`  |

硬件环境部署可使用`NVIDIA`提供的`SDK Manager`工具，宿主机磁盘空间和内存一定要大，否则可能引发各种错误，推荐配置如下：
* 磁盘`>150GB`（可用不小于120GB）
* 内存`>=6GB`

## segment-anything模型版本
> 可在[页面](https://github.com/FasterSegmentAnything/segment-anything#model-checkpoints)下载

| 类型 | 简述 |
| :-- | :-- |
| 名称 | `segment-anything` |
| 版本 | `SAM_VIT_H 4b8939` |
| 下载地址| [链接](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)  | 

# segment-anything安装

```shell
# 依赖项安装
pip3 install opencv-python pycocotools matplotlib onnxruntime onnx

# 添加环境变量
# 向/etc/profile、~/.bashrc末尾添加"export PATH=/home/jetson/.local/bin:$PATH"
# 通过source /etc/profile && source ~/.bashrc使其生效

# 安装segment-anything
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip3 install -e .
```

# 关于

* [论文原文](files/related/paper.pdf)
* [官方网站](https://segment-anything.com/)
* [模型下载地址](https://github.com/FasterSegmentAnything/segment-anything#model-checkpoints)
* 作者
  * [yutian](https://www.aflyingfish.top/)