import time,os
import argparse

import cv2
import numpy as np

from deploy.predictor import SamPredictor
import models

# 输入
# image="demo1.jpg"
image="demo2.png"
# image="demo3.jpg"
mymodel=models.SAM_VIT_B            # 使用的模型
enableWarmup=False                  # 是否启动系统预热
output=""

# ------------------------------------ 分割线 ------------------------------------------- #

# 初始化SamPredictor
# "cuda"表示用GPU运算，"cpu"表示用CPU。系统预热能显著提升运行速度，但当前设备启动预热会导致硬件内存不足，因此示例代码预热次数为0次。实际预热推荐>=3次
predictor = SamPredictor(vit_model_path=mymodel.image_vit_model_path,decoder_model_path=mymodel.onnx_model_path,device="cuda",warmup_epoch=3 if enableWarmup else 0)    

# 从inputs文件夹读入待识别的图片
print("load input:",os.path.join("inputs",image))
img = cv2.imread(os.path.join("inputs",image))

# 分析图片特征（内部使用vit模型推理，内存需求大，耗时长）
predictor.register_image(img)           

# 用户构建输入
points = []       # 识别时用户提示的点集
boxes = []        # 识别时用户提示的框集

# 一些辅助数据结构
box_point = []
img_copy = img.copy()
get_first_box_point = False     # 用于读取box两点时的flag

# 由两个点标记一个框时，计算左上角和右下角点的坐标
def change_box(box):
    x1 = min(box[0], box[2])
    y1 = min(box[1], box[3])
    x2 = max(box[0], box[2])
    y2 = max(box[1], box[3])
    return [x1, y1, x2, y2]

# 识别+显示
def draw_circle(event, x, y, flags, param):
    global img, get_first_box_point, box_point
    if event == cv2.EVENT_LBUTTONDOWN and not get_first_box_point and flags!=cv2.EVENT_FLAG_CTRLKEY+cv2.EVENT_LBUTTONDOWN:
        # ctrl+左键
        print("Add point:", (x, y))
        points.append([x, y])

        # 计算一次推理，输入点和框，生成cover图
        mask = predictor.get_mask(points, [1]*len(points), boxes if boxes != [] else None)

        # 推理结果解码成图片
        mask = mask["masks"][0][0][:, :, None]
        cover = np.ones_like(img) * 255
        cover = cover * mask
        cover = np.uint8(cover)

        # cv2.addWeighted即将`原图`与`cover图`合为一张图
        # 合成时各像素通道值的计算公式：dst(I) = saturate(src1(I) ∗ 0.6 + src2(I) ∗ 0.4 + 0)
        # 此处0.6和0.4可修改，一般建议和为1.0
        img = cv2.addWeighted(img_copy, 0.6, cover, 0.4, 0)

        for b in boxes:
            # 绘制绿色的框
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            # 红点绘制框左上角的点
            cv2.circle(img, (b[0], b[1]), 5, (0, 0, 255), -1)
        for p in points:
            # 蓝色绘制框左上角的点
            cv2.circle(img, (p[0], p[1]), 5, (255, 0, 0), -1)

    elif  event==cv2.EVENT_LBUTTONDOWN and flags==cv2.EVENT_FLAG_CTRLKEY+cv2.EVENT_LBUTTONDOWN:
        # ctrl+左键
        if not get_first_box_point:
            box_point.append(x)
            box_point.append(y)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            get_first_box_point = True
        else:
            box_point.append(x)
            box_point.append(y)
            box_point = change_box(box_point)
            boxes.append(box_point)
            print("add box:", box_point)

            # 计算一次推理，输入点和框，生成cover图
            mask = predictor.get_mask(points if points != [] else None, [1 for i in range(len(points))] if points != [] else None, boxes)

            # 推理结果解码成图片
            mask = mask["masks"][0][0][:, :, None]
            cover = np.ones_like(img) * 255
            cover = cover * mask
            cover = np.uint8(cover)

            # cv2.addWeighted即将`原图`与`cover图`合为一张图
            # 合成时各像素通道值的计算公式：dst(I) = saturate(src1(I) ∗ 0.6 + src2(I) ∗ 0.4 + 0)
            # 此处0.6和0.4可修改，一般建议和为1.0
            img = cv2.addWeighted(img_copy, 0.6, cover, 0.4, 0)

            get_first_box_point = False
            box_point = []

        for b in boxes:
            # 绿色绘制框边线
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            # 红点绘制框左上角的点
            cv2.circle(img, (b[0], b[1]), 5, (0, 0, 255), -1)
            # 红点绘制框右下角的点
            cv2.circle(img, (b[2], b[3]), 5, (0, 0, 255), -1)
        for p in points:
            # 蓝色绘制框左上角的点
            cv2.circle(img, (p[0], p[1]), 5, (255, 0, 0), -1)

    # 刷新图
    cv2.imshow("image", img)

# UI总控
cv2.namedWindow("image")
cv2.setMouseCallback('image', draw_circle)
print("press `q` to end process, and `r` to clear inputs.")
while True:
    key = cv2.waitKey()
    if key & 0xff == ord("q"):
        # 按q退出
        if output != "":
            cv2.imwrite(os.path.join("outputs",output), img)
        print("end process.")
        break
    elif key & 0xff == ord("r"):
        # 按r重置
        points = []
        boxes = []
        box_point = []
        img = cv2.imread(os.path.join("inputs",image))
        print("reset all inputs")
cv2.destroyAllWindows()
