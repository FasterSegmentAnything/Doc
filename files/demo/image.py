import cv2

def read_image(image_jpg_path):
    image = cv2.imread(image_jpg_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
