import numpy as np
import cv2
from imhblpce import imhblpce

def test_im2dhisteq_with_param():
    image_name = '../assets/Countryside.jpg'
    image = cv2.imread(image_name)

    # converts rgb image to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_image = hsv_image[:, :, 2]
    v_image_hblpce = imhblpce(v_image)
    
    # np.save(f'{image_name}-hblpce', v_image_hblpce)

    v_image_hblpce_cmpr = np.load(f'{image_name}-hblpce.npy')
    assert np.all(v_image_hblpce == v_image_hblpce_cmpr)

    
    