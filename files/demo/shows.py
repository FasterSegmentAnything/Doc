import numpy as np
import cv2
import matplotlib.pyplot as plt
from from_pytorch_model import sam_model_registry
from image import read_image

def show_mask(mask, ax=None):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image)
    cv2.imwrite("outputs/mask_output.png", mask_image)
    
def show_points(coords, labels, marker_size=375, ax=None):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax=None):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def show_image(image_jpg_path):
    img = read_image(image_jpg_path)
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # plt.axis('on')
    # plt.show()
    cv2.imwrite("outputs/image_output.png", img)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

    plt.savefig("outputs/anns_output.png")
    # cv2.imwrite("outputs/image_output.png", img)  # 无法显示

    plt.show()