目录
- [一、说明](#一说明)
- [二、实验环境](#二实验环境)
  - [1、硬件：](#1硬件)
  - [2、软件：](#2软件)
  - [3、segment-anything模型版本](#3segment-anything模型版本)
- [三、segment-anything安装](#三segment-anything安装)
- [四、使用方式](#四使用方式)
- [五、关于](#五关于)

# 一、说明

> Github仓库commit版本：6fdee8f2727f4506cfbbe553e23b895e27956588

解析[segment-anything](https://segment-anything.com/)项目。

# 二、实验环境

> 无特殊指定的情况下，本文档默认指代环境如下：

## 1、硬件：

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

## 2、软件：

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
| pytorch | 自构建`2.1.0a0+gitc9f4f01` |
| onnxruntime-GPU(python版) |  自构建`1.12.1` |
| onnxruntime-GPU(C++版) |  自构建`1.12.1` |
| opencv-python | `4.7.0` |
| pycocotools | `2.0.6` |
| numpy | `1.24.3` |
| matplotlib | `3.1.2` |
| protobuf | `3.20.3` |
| typing-extensions | `4.5.0` |
| onnx | `1.13.1` |
| torchvision | 自构建`0.16.0a0+01b9faa` |

> onnxruntime安装：
> 
> 编译：./build.sh  --skip_tests --use_cuda --config Release --build_shared_lib --parallel 6  --build_wheel  --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda
>
> * C++版本安装：添加环境变量
>
> * python版本安装：cd build/Linux/Release/dist && pip3 install ./onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl 

> pytorch安装参考：https://blog.csdn.net/github_34897521/article/details/105123812

硬件环境部署可使用`NVIDIA`提供的`SDK Manager`工具，宿主机磁盘空间和内存一定要大，否则可能引发各种错误，推荐配置如下：
* 虚拟机建议采用`Ubuntu>=18.04`
* `apt-get`的source.list尽可能采用默认，测试中阿里源会导致error
* 磁盘`>150GB`（可用不小于120GB）
* 内存`>=6GB`

## 3、segment-anything模型版本
> 可在[页面](https://github.com/FasterSegmentAnything/segment-anything#model-checkpoints)下载

| 类型 | 版本 | 文件 | 下载链接 | 状态 |
| :-- | :-- | :-- | :-- | :-- |
| 名称 | `segment-anything` |
| `SAM_VIT_H` | `SAM_VIT_H 4b8939` | [本地文件](files/pytorch_model/sam_vit_h_4b8939.pth) | [链接](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)  | 内存需求过大，暂时弃用 |
| `SAM_VIT_B` | `SAM_VIT_B 01ec64` | [本地文件](files/pytorch_model/sam_vit_b_01ec64.pth) | [链接](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)  | 使用中 |

# 三、segment-anything安装

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

# 四、使用方式

[文档](usage.md)

# 五、关于

* [论文原文](files/related/paper.pdf)
* [官方网站](https://segment-anything.com/)
* [模型下载地址](https://github.com/FasterSegmentAnything/segment-anything#model-checkpoints)
* 作者
  * [yutian](https://www.aflyingfish.top/)
