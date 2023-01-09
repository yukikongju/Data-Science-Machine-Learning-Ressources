# Problem: We want to find all occurences of an object in an image. 
# However, sometimes, when trying to find all locations under a given 
# threshold, we find that there are multiple rectangle for the same object. 
# Our goal is to only keep one rectangle for each object. We eliminate 
# redundant rectangles

# Notes on `groupRectangle`: 
# - groupThreshold: 
# - eps: relative difference between sides of rectangles


import cv2 as cv
import numpy as np
import pathlib
import os

def find_click_positions(template_path, target_path, threshold=0.5, debug_mode=None):
    # 1. load template and target images
    template_img = cv.imread(template_path, cv.IMREAD_UNCHANGED)
    target_img = cv.imread(target_path, cv.IMREAD_UNCHANGED)


    # 2. perform template matching
    method = cv.TM_SQDIFF_NORMED
    result = cv.matchTemplate(target_img, template_img, method)
    #  print(len(result))

    # 3. find all locations with respective rectangles 
    locations = np.where(result <= threshold)
    locations = list(zip(*locations[::-1]))
    #  print(len(locations))

    template_width = template_img.shape[1]
    template_height = template_img.shape[0]

    rectangles = []
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), template_width, template_height]
        rectangles.append(rect)
        rectangles.append(rect)
    rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

    # 4. find click positions
    points = []
    if len(rectangles):
        line_color = (255, 255, 255)
        line_type = cv.LINE_4
        marker_color = (255, 0 , 255)
        marker_type = cv.MARKER_CROSS

        for (x, y, w, h) in rectangles:
            # determine center position
            center_x = x + int(w/2)
            center_y = y + int(h/2)

            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv.rectangle(target_img, top_left, bottom_right, color=line_color, 
                             lineType=line_type, thickness=2)
            elif debug_mode == 'points':
                cv.drawMarker(target_img, (center_x, center_y), 
                              color=marker_color, markerType=marker_type, 
                              markerSize=40, thickness=2)

            if debug_mode:
                cv.imshow('', target_img)
                cv.waitKey()

    return points


def main():
    image_base_path = os.path.join(pathlib.Path(__file__).parents[1], 'images/')
    template_path = os.path.join(image_base_path, 'template_rod.png')
    target_path = os.path.join(image_base_path, 'pond1.png')
    threshold = 0.105
    points = find_click_positions(template_path, target_path, threshold=threshold, debug_mode='rectangles')
    points = find_click_positions(template_path, target_path, threshold=threshold, debug_mode='points')
    print(points)
    

if __name__ == "__main__":
    main()
