- [一、模型导出为ONNX格式](#一模型导出为onnx格式)
  - [1. 准备文件](#1-准备文件)
  - [2. 下载模型](#2-下载模型)
  - [3. 环境准备](#3-环境准备)
  - [4. 导出`onnx`模型](#4-导出onnx模型)
- [二、模型运行示例(基于onnxruntime)](#二模型运行示例基于onnxruntime)
  - [1. 完整应用示例](#1-完整应用示例)
  - [2. 识别框（以全图为例）](#2-识别框以全图为例)


# 一、模型导出为ONNX格式

> 下载的模型为pytorch模型

## 1. 准备文件

```shell
mkdir -p ~/FasterSegmentAnything
cd ~/FasterSegmentAnything

git clone https://github.com/facebookresearch/segment-anything.git
git clone https://github.com/FasterSegmentAnything/Doc.git

mkdir -p Models/onnx
mkdir -p Models/raw_model

# 增加导出`image vit`模型的脚本
cp -r Doc/scripts/ImageEncoderOnnxModel.py segment-anything/segment_anything/utils/
cp -r Doc/scripts/export_image_onnx.py segment-anything/scripts/
```

形成的目录结构如下：

```shell
FasterSegmentAnything       # 位于`~/`所指代的目录（即home目录）
├── Doc        #（为Doc项目）
│   └── 省略
├── Models
│   ├── onnx
│   └── raw_model
└── segment-anything        #（为segment-anything项目）
    └── 省略
```

## 2. 下载模型

> 参考[小节](https://github.com/FasterSegmentAnything/Doc#3segment-anything%E6%A8%A1%E5%9E%8B%E7%89%88%E6%9C%AC)

```shell
cd ~/FasterSegmentAnything/Models/raw_model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

形成的目录结构如下：

```shell
FasterSegmentAnything       # 位于`~/`所指代的目录（即home目录）
├── Doc        #（为Doc项目）
│   └── 省略
├── Models
│   ├── onnx
│   └── raw_model
│       └── sam_vit_b_01ec64.pth
└── segment-anything        #（为segment-anything项目）
    └── 省略
```


## 3. 环境准备

```shell
pip3 install opencv-python pycocotools matplotlib onnxruntime onnx
pip3 install torch torchvision opencv-contrib-python
```

## 4. 导出`onnx`模型
> 注意：官方仓库仅支持导出`decoder`模型，导出`image vit`模型的代码为本文档新增。如果在后续版本中遇到问题，请使用本文档所使用的仓库版本。

* 导出`decoder`模型

  ``` shell
  cd ~/FasterSegmentAnything/segment-anything/
  python3 scripts/export_onnx_model.py --checkpoint ../Models/raw_model/sam_vit_b_01ec64.pth --model-type vit_b --output ../Models/onnx/SAM-VITb.onnx
  # 此处model-type取决于下载的模型，可选项为：`default`、`vit_h`、`vit_l`、`vit_b`。 vit_b对内存需求最小，`default`等价于`vit_h`
  ```

* 导出`image vit`模型

  ``` shell
  cd ~/FasterSegmentAnything/segment-anything/
  python3 scripts/export_image_onnx.py --checkpoint ../Models/raw_model/sam_vit_b_01ec64.pth --model-type vit_b --output ../Models/onnx/SAM-image-VITb.onnx
  # 此处model-type取决于下载的模型，可选项为：`default`、`vit_h`、`vit_l`、`vit_b`。 vit_b对内存需求最小，`default`等价于`vit_h`
  ```

此时目录结构如下：

```shell
FasterSegmentAnything       # 位于`~/`所指代的目录（即home目录）
├── Doc                     #（为Doc项目）
│   └── 省略
├── Models
│   ├── onnx
│   │   ├── SAM-image-VITb.onnx
│   │   └── SAM-VITb.onnx
│   └── raw_model
│       └── sam_vit_b_01ec64.pth
└── segment-anything         #（为segment-anything项目）
    └── 省略
```

将生成的模型复制到Doc项目下：

```shell
mkdir -p ~/FasterSegmentAnything/Doc/files/onnx
mkdir -p ~/FasterSegmentAnything/Doc/files/pytorch_model

cd ~/FasterSegmentAnything
cp -r Models/onnx/*.onnx ~/FasterSegmentAnything/Doc/files/onnx/
cp -r Models/raw_model/*.pth ~/FasterSegmentAnything/Doc/files/pytorch_model/
```

# 二、模型运行示例(基于onnxruntime)

> 需要安装onnxruntime-GPU版运行环境，可参考[文档](README.md)。为了增强文档的说明性，本文档直接对各项基础功能撰写示例代码。
>
> 以下代码需要展示绘图，需要开启开发板的`X11-forward`，并在客户机开启桌面转发支持（推荐使用`MobaXterm`）

## 1. 完整应用示例

运行[示例代码](demo1.py)，按`q`退出，按`r`重置。

```shell
cd ~/FasterSegmentAnything/Doc && python3 demo1.py
```

功能：
* 输入：一张图片
  * 可通过修改示例代码中`image="demo3.jpg"`调整输入
* 事件
  * 输入`点`
    * 用`鼠标左键`点击图片任意位置可输入一个`点（坐标）`
  * 输入`框`
    * 用`ctrl+鼠标左键`在图片任意位置点击两次（分别指代框的`左上角`和`右下角`），可输入一个`框`。
* 规则
  * 每成功输入一个`点`或者一个`框`，将运行一次`全新的识别`，并显示新一轮输出
  * 系统将历史累计输入的`点`、`框`一起作为`用户提示内容`。如果您想清除这些历史输入，可按`r`重置
* 输出：分割出`用户提示内容`所指代的内容

## 2. 识别框（以全图为例）

运行[示例代码](demo2.py)：

```shell

```