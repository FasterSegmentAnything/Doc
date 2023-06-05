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
cv2.imwrite(os.path.join("outputs",output), img)

cv2.waitKey()                                               # 程序中断等待