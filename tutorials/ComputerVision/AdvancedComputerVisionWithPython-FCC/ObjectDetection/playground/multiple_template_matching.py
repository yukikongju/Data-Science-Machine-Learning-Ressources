# Problem: We want to find all instance of an object inside an image given 
# a template image and an image

import cv2 as cv
import os
import numpy as np
import pathlib


# 1. Load template and target images
image_base_path = os.path.join(pathlib.Path(__file__).parents[1], 'images/')
template_path = os.path.join(image_base_path, 'template_rod.png')
target_path = os.path.join(image_base_path, 'pond1.png')

template = cv.imread(template_path, cv.IMREAD_UNCHANGED)
target = cv.imread(target_path, cv.IMREAD_UNCHANGED)

# 2. Perform template matching
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
result = cv.matchTemplate(target, template, cv.TM_SQDIFF_NORMED)

# 3. find locations above threshold
threshold = 0.105
locations = np.where(result <= threshold)
locations = list(zip(*locations[::-1]))
#  print(locations)

# 4. draw rectangle for each locations
line_color = (255, 255, 255)
line_type = cv.LINE_4
if locations: 
    for location in locations:
        top_left = location
        bottom_right = (location[0] + template.shape[1],
                        location[1] + template.shape[0])
        cv.rectangle(target, top_left, bottom_right, line_color, line_type)
    cv.imshow('', target)
    cv.waitKey()

print(len(locations))
