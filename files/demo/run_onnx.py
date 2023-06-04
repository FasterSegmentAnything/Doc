import cv2
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from from_pytorch_model import GetSamModel
from shows import *
import image,time
import matplotlib
import os,time
import models

matplotlib.use('tkagg')     # 可用mobaXterm远程显示绘图
os.chdir(os.path.dirname(os.path.abspath(__file__)))

start_time=time.time()
model=models.SAM_VIT_B
input_image="inputs/demo3.jpg"
img=image.read_image(input_image)

plt.figure(figsize=(20,20))
plt.imshow(img)
plt.axis('on')
# plt.show()

ort_session = onnxruntime.InferenceSession(model.onnx_model_path, providers=['CUDAExecutionProvider'])
sam = sam_model_registry[model.type](checkpoint=model.pytorch_model_path)
sam.to(device='cuda')
predictor = SamPredictor(sam)

for _ in range(1):
    # transform
    start_time=time.time()
    predictor.set_image(img)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    print(image_embedding.shape)
    transform_cost=time.time()-start_time

    input_point = np.array([[500, 375]], dtype=np.float32)
    input_label = np.array([1], dtype=np.float32)
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]]).astype(np.float32)], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    nnx_coord = predictor.transform.apply_coords(onnx_coord, img.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(img.shape[:2], dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    total_time=time.time()-start_time
    print("=>transfer cost(s):",transform_cost)
    print("onnx inference cost(s):",total_time-transform_cost)
    print("total cost(s):",total_time)

masks = masks > predictor.model.mask_threshold
print(masks.shape)
print("-----------\n",masks)
plt.figure(figsize=(10,10))
plt.imshow(img)
show_mask(masks, ax=plt.gca())
show_points(input_point, input_label, ax=plt.gca())
plt.axis('off')
plt.show() 