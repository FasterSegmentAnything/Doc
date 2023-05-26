> segment-anything使用说明

- [一、模型导出为ONNX格式](#一模型导出为onnx格式)
  - [1. 模型准备](#1-模型准备)
  - [2. 环境准备](#2-环境准备)
  - [3. 导出过程](#3-导出过程)
  - [4. 脚手架](#4-脚手架)
- [二、模型运行](#二模型运行)

> 请确保实现下载了segment-anything项目代码

相关目录结构如下：

```shell
FasterSegmentAnything       # 位于`~/`所指代的目录（即home目录）
├── Models
│   ├── onnx
│   └── raw_model
│       └── sam_vit_h_4b8939.pth
└── segment-anything        #（为segment-anything项目）
    └── 省略
```

# 一、模型导出为ONNX格式

> 下载的模型为pytorch模型

## 1. 模型准备

下载模型，参考[小节](https://github.com/FasterSegmentAnything/Doc#3segment-anything%E6%A8%A1%E5%9E%8B%E7%89%88%E6%9C%AC)

## 2. 环境准备

```shell
pip3 install opencv-python pycocotools matplotlib onnxruntime onnx
pip3 install torch torchvision opencv-contrib-python
```

## 3. 导出过程

``` shell
cd ~/FasterSegmentAnything/segment-anything/
python3 scripts/export_onnx_model.py --checkpoint ../Models/raw_model/sam_vit_h_4b8939.pth --model-type default --output ../Models/onnx/SAM-VITh.onnx
python3 scripts/export_onnx_model.py --checkpoint ../Models/raw_model/sam_vit_b_01ec64.pth --model-type vit_b --output ../Models/onnx/SAM-VITb.onnx
# 此处model-type取决于下载的模型，可选项为：`default`、`vit_h`、`vit_l`、`vit_b`

```
> 输出(示例)
> Loading model...
> 
> Exporting onnx model to ../Models/onnx/SAM-VITh.onnx...
> 
> ================ Diagnostic Run torch.onnx.export version 2.0.1 ================
> 
> verbose: False, log level: Level.ERROR
> 
> ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================
> 
> Model has successfully been run with ONNXRuntime.

此时目录结构如下：

```shell
FasterSegmentAnything       # 位于`~/`所指代的目录（即home目录）
├── Models
│   ├── onnx
│   │   ├── SAM-VITb.onnx
│   │   └── SAM-VITh.onnx
│   └── raw_model
│       ├── sam_vit_b_01ec64.pth
│       └── sam_vit_h_4b8939.pth
└── segment-anything        #（为segment-anything项目）
    └── 省略
```

## 4. 脚手架

[导出onnx模型下载](onnx/SAM-VITh.onnx)

# 二、模型运行

