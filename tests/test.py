import numpy as np
import cv2
from imhblpce import imhblpce

def imresize(img, wr=500, hr=None): # This is just for imshow-ing images with titles
    [ h, w, d] = img.shape
    hr = (h*wr)//w if not hr else hr
    img_resized = cv2.resize(img, dsize=(wr, hr))
    return img_resized

def main():
    image_name = '../assets/Countryside.jpg'
    image = cv2.imread(image_name)

    # converts rgb image to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_image = hsv_image[:, :, 2]
    v_image_hblpce = imhblpce(v_image)
    
    hsv_image_hblpce = hsv_image.copy()
    hsv_image_hblpce[:, :, 2] = v_image_hblpce
    image_hblpce = cv2.cvtColor(hsv_image_hblpce, cv2.COLOR_HSV2BGR)

    # This is just for imshow-ing images with titles

    cv2.imshow('Original Image', imresize(image))
    cv2.imshow('HBLPCE-d Image', imresize(image_hblpce))
    cv2.waitKey(0)
    
    
if __name__ == '__main__': main()