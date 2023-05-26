import cv2
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from from_pytorch_model import GetSamModel
from shows import *
import image
import matplotlib
import os,time
import models

matplotlib.use('tkagg')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

start_time=time.time()
model=models.SAM_VIT_B
input_image="inputs/demo1.jpg"
img=image.read_image(input_image)
# ort_session = onnxruntime.InferenceSession(model.model_path)

# # ---------------使用pytorch推理（prompts）-------------------
# sam = GetSamModel(pytorch_model_path)
# predictor = SamPredictor(sam)
# predictor.set_image(image.read_image(input_image))
# # image_embedding = predictor.get_image_embedding().cpu().numpy()
# # print(image_embedding.shape)
# masks, _, _ = predictor.predict()
# show_anns(masks)
## ------------------------------------------------------------

# ---------------使用pytorch推理（prompts）-------------------
sam = GetSamModel(model.pytorch_model_path,"vit_b")
# image_embedding = predictor.get_image_embedding().cpu().numpy()
# print(image_embedding.shape)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)
print("time cost",time.time()-start_time,"s")
print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(img)
plt.savefig("outputs/raw.png")
show_anns(masks)

plt.show()
## ------------------------------------------------------------