- [一、模型导出为ONNX格式](#一模型导出为onnx格式)
  - [1. 准备文件](#1-准备文件)
  - [2. 下载模型](#2-下载模型)
  - [3. 环境准备](#3-环境准备)
  - [3. 导出`onnx`模型](#3-导出onnx模型)
- [二、模型运行示例(基于onnxruntime)](#二模型运行示例基于onnxruntime)
  - [1、 分割全图](#1-分割全图)


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

## 3. 导出`onnx`模型
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
├── Doc        #（为Doc项目）
│   └── 省略
├── Models
│   ├── onnx
│   │   ├── SAM-image-VITb.onnx
│   │   └── SAM-VITb.onnx
│   └── raw_model
│       └── sam_vit_b_01ec64.pth
└── segment-anything        #（为segment-anything项目）
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

## 1、 分割全图

[示例代码](demo1.py)

功能：
* 输入：一张图片
* 输出：整张图片的掩码图

