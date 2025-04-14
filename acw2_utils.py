import os
import cv2
from ultralytics import YOLO
#from acw2_utils import CLASS_MAPPING, normalize_bbox 



# Dictionary mapping class index to (sign_number, sign_name)
CLASS_MAPPING = {
    0: (1, "Roundabout"),
    1: (4, "Traffic lights"),
    2: (5, "Roadworks"),
    3: (13, "No entry"),
    4: (16, "30MPH"),
    5: (19, "National speed limit")
}

def normalize_bbox(xyxy, image_shape):
    x1, y1, x2, y2 = xyxy
    h, w = image_shape[:2]
    x_centre = ((x1 + x2) / 2) / w
    y_centre = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    return x_centre, y_centre, width, height