[![PyPI Latest Release](https://img.shields.io/pypi/v/imhblpce.svg)](https://pypi.org/project/imhblpce/)
[![Package Status](https://img.shields.io/pypi/status/imhblpce.svg)](https://pypi.org/project/imhblpce/)
[![Downloads](https://pepy.tech/badge/imhblpce)](https://pepy.tech/project/imhblpce)
[![License](https://img.shields.io/pypi/l/imhblpce.svg)](https://github.com/Mamdasn/imhblpce/blob/main/LICENSE)
![Repository Size](https://img.shields.io/github/repo-size/mamdasn/imhblpce)


# imhblpce
This module attempts to enhance contrast of a given image by employing a method called HBLPCE [Histogram-Based Locality-Preserving Contrast Enhancement]. This method enhances contrast of an image through equalizing its histogram, while keeping an eye on histogram's general shape, to conserve overall brightness and prevent excessive enhancement of the image.  

You can access the article that came up with this method [here](https://www.researchgate.net/publication/272424815_Histogram-Based_Locality-Preserving_Contrast_Enhancement).  

Through formulating their approach, a minimization problem is introduced and solved using cvxpy library in python.

## Installation

Run the following to install:

```python
pip install imhblpce
```

## Usage

```python
import numpy as np
import cv2
from imhblpce import imhblpce

def imresize(img, wr=500, hr=None): # This is just for imshow-ing images with titles
    [ h, w, d] = img.shape
    hr = (h*wr)//w if not hr else hr
    img_resized = cv2.resize(img, dsize=(wr, hr))
    return img_resized

def main():
    image_name = 'assets/Countryside.jpg'
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
```  
Or  
```Bash
imhblpce --input 'Countryside.jpg' --output 'Countryside-imhblpce.jpg'
```  

## Showcase
This is a sample image
![Countryside.jpg Image](https://raw.githubusercontent.com/Mamdasn/imhblpce/main/assets/Countryside.jpg "Countryside.jpg Image")
The sample image enhanced by HBLPCE method
![Countryside-imhblpce.jpg Image](https://raw.githubusercontent.com/Mamdasn/imhblpce/main/assets/Countryside-imhblpce.jpg "Countryside-imhblpce.jpg")
