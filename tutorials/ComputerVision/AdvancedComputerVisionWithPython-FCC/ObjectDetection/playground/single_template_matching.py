import numpy as np
import cv2 as cv
import os
import pathlib
import matplotlib.pyplot as plt


# 1. Load template and target images
image_base_path = os.path.join(pathlib.Path(__file__).parents[1], 'images/')
template_path = os.path.join(image_base_path, 'template_rod.png')
target_path = os.path.join(image_base_path, 'pond1.png')

template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
target = cv.imread(target_path, cv.IMREAD_GRAYSCALE)

# 2. Perform Single Template Matching
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
result = cv.matchTemplate(target, template, cv.TM_CCORR_NORMED)
#  plt.imshow(result, cmap = 'gray')
#  plt.show()

# 3. Find best match 
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
cv.rectangle(target, top_left, bottom_right, 255, 2)
plt.imshow(target, cmap = 'gray')
plt.show()


