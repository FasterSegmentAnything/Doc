- [一、模型导出为ONNX格式](#一模型导出为onnx格式)
  - [1. 准备文件](#1-准备文件)
  - [2. 下载模型](#2-下载模型)
  - [3. 环境准备](#3-环境准备)
  - [4. 导出`onnx`模型](#4-导出onnx模型)
- [二、模型运行示例(基于onnxruntime)](#二模型运行示例基于onnxruntime)
  - [1. 完整应用示例](#1-完整应用示例)
  - [2. 示例代码](#2-示例代码)


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
    * 用`ctrl+鼠标左键`在图片任意位置点击两次（指代框的对角点），可输入一个`框`。
* 规则
  * 每成功输入一个`点`或者一个`框`，将运行一次`全新的识别`，并显示新一轮输出
  * 系统将历史累计输入的`点`、`框`一起作为`用户提示内容`。如果您想清除这些历史输入，可按`r`重置
* 输出：分割出`用户提示内容`所指代的内容

## 2. 示例代码
> 内容同[示例代码](demo2.py)

```python
import time,os
import argparse

import cv2
import numpy as np

from deploy.predictor import SamPredictor
import models

# 输入
image="demo2.png"                   # 表示识别inputs文件夹下的demo2.png
mymodel=models.SAM_VIT_B            # 使用的模型
enableWarmup=False                  # 是否启动系统预热
output="demo2_output.png"           # 本地存储标记后的图片

# 初始化SamPredictor
# "cuda"表示用GPU运算，"cpu"表示用CPU。系统预热能显著提升运行速度，但当前设备启动预热会导致硬件内存不足，因此示例代码预热次数为0次。实际预热推荐>=3次
predictor = SamPredictor(vit_model_path=mymodel.image_vit_model_path,decoder_model_path=mymodel.onnx_model_path,device="cuda",warmup_epoch=3 if enableWarmup else 0)    

# 从inputs文件夹读入待识别的图片
print("load input:",os.path.join("inputs",image))
img = cv2.imread(os.path.join("inputs",image))
img_copy = img.copy()

# 用户构建输入
points = [[100,200],[150,300]]                              # 识别时用户提示的点集，即(100,200)、(150,300)两个点。可以为[]
boxes = [[0,0,300,300]]                                     # 识别时用户提示的框集，即左上角为(0,0)，右下角为(300,300)。可以为[]

predictor.register_image(img)                               # 分析图片特征（内部使用vit模型推理，内存需求大，耗时长）
mask = predictor.get_mask(points, [1]*len(points), boxes)   # 识别解码（内部使用decode模型推理，内存需求一般，耗时相对较少）

# 对推理结果解码
mask = mask["masks"][0][0][:, :, None]
cover = np.ones_like(img) * 255                             # 生成一张与原图同样尺寸的纯白色图片
cover = cover * mask                      
cover = np.uint8(cover)                                     # 生成识别图层

# cv2.addWeighted即将`原图`与`cover图`合为一张图
# 合成时各像素通道值的计算公式：dst(I) = saturate(src1(I) ∗ 0.6 + src2(I) ∗ 0.4 + 0)
# 此处0.6和0.4可修改，一般建议和为1.0
img = cv2.addWeighted(img_copy, 0.6, cover, 0.4, 0)         # 将识别图层与原图合并

# 绘制出所有的点和框
for b in boxes:
    # 绿色绘制框边线
    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
for p in points:
    # 蓝色绘制框左上角的点
    cv2.circle(img, (p[0], p[1]), 5, (255, 0, 0), -1)


cv2.imshow("image", img)                                    # 展示最终结果
cv2.imwrite(os.path.join("outputs",output), img)            # 内容存储到outputs文件夹下

cv2.waitKey()                                               # 程序中断等待
```  