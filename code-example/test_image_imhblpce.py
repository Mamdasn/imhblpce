from imhblpce import imhblpce
import numpy as np
import cv2
import os

filename = "assets/Plane.jpg"
name, ext = os.path.splitext(filename)
image = cv2.imread(filename)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_v = image_hsv[:, :, 2].copy()

image_v_imhblpce = imhblpce(image_v)

image_hsv[:, :, 2] = image_v_imhblpce.copy()
image_imhblpce = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

cv2.imwrite(f"{name}-imhblpce{ext}", image_imhblpce)
