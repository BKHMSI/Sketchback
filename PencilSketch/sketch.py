import cv2 as cv
import numpy as np
import os
from filters import PencilSketch


for file in os.listdir('ZuBuD'):
    file_path = os.path.join('ZuBuD', file)
    img = cv.imread(file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    print (img.shape)
    height, width, channels = img.shape
    if (height, width, channels) == (240,320,3):
	    img = np.resize(img, (480, 640, 3))
    pencil = PencilSketch(width, height)
    sketch = pencil.render(img)
    write_path = os.path.join('ZuBuD_Sketch', file)
    cv.imwrite(write_path, sketch)
    