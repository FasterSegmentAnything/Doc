# 显示全图掩码

import cv2
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
from segment_anything.utils.onnx import SamOnnxModel
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from from_pytorch_model import GetSamModel
from shows import *
import image
import matplotlib
import os,time
import models

matplotlib.use('tkagg')     # 可用mobaXterm远程显示绘图
os.chdir(os.path.dirname(os.path.abspath(__file__)))
model=models.SAM_VIT_B

input_image="inputs/demo1.jpg"

start_time=time.time()

## 开始推理
img=image.read_image(input_image)
sam = GetSamModel(model.pytorch_model_path,model.type)
sam.to(device='cuda')
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)
## 结束推理


## 效果解析
print("time cost",time.time()-start_time,"s")
print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(img)
show_anns(masks)