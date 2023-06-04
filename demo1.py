import time,os
import argparse

import cv2
import numpy as np

from deploy.predictor import SamPredictor
import models

# 输入
image="demo3.jpg"
mymodel=models.SAM_VIT_B            # 使用的模型
enableWarmup=False                  # 是否启动系统预热
output=""

# ------------------------------------ 分割线 ------------------------------------------- #

# 初始化SamPredictor
predictor = SamPredictor(vit_model_path=mymodel.image_vit_model_path,decoder_model_path=mymodel.onnx_model_path,device="cuda",warmup_epoch=3 if enableWarmup else 0)

# 读入输入图片
print("load input:",os.path.join("inputs",image))
img = cv2.imread(os.path.join("inputs",image))

# 分析图片特征（内部使用vit模型推理）
predictor.register_image(img)           

points = []
boxes = []
box_point = []
img_copy = img.copy()
get_first_box_point = False


def change_box(box):
    x1 = min(box[0], box[2])
    y1 = min(box[1], box[3])
    x2 = max(box[0], box[2])
    y2 = max(box[1], box[3])
    return [x1, y1, x2, y2]


def draw_circle(event, x, y, flags, param):
    global img, get_first_box_point, box_point
    if event == cv2.EVENT_LBUTTONDOWN and not get_first_box_point:
        print("Add point:", (x, y))
        points.append([x, y])

        # 计算一次推理，输入点和框
        mask = predictor.get_mask(points, [1]*len(points), boxes if boxes != [] else None)

        mask = mask["masks"][0][0][:, :, None]
        cover = np.ones_like(img) * 255
        cover = cover * mask
        cover = np.uint8(cover)
        img = cv2.addWeighted(img_copy, 0.6, cover, 0.4, 0)

        for b in boxes:
            # 绘制绿色的框
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            # 红点绘制框左上角的点
            cv2.circle(img, (b[0], b[1]), 5, (0, 0, 255), -1)
        for p in points:
            # 蓝色绘制框左上角的点
            cv2.circle(img, (p[0], p[1]), 5, (255, 0, 0), -1)

    elif event == cv2.EVENT_RBUTTONDOWN:
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

            mask = predictor.get_mask(points if points != [] else None,
                                      [1 for i in range(len(points))] if points != [] else None,
                                      boxes)

            mask = mask["masks"][0][0][:, :, None]
            cover = np.ones_like(img) * 255
            cover = cover * mask
            cover = np.uint8(cover)
            img = cv2.addWeighted(img_copy, 0.6, cover, 0.4, 0)

            get_first_box_point = False
            box_point = []

        for b in boxes:
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


cv2.namedWindow("image")
cv2.setMouseCallback('image', draw_circle)
key = cv2.waitKey()
if key & 0xff == ord("s"):
    if output != "":
        cv2.imwrite(os.path.join("outputs",output), img)
cv2.destroyAllWindows()
