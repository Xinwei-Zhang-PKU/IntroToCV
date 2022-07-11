# https://github.com/adityaintwala/Harris-Corner-Detection/blob/master/find_harris_corners.py
import numpy as np
import cv2

def harris_corner(input_image, alpha, window_size, threshold):
    corner_list = []
    output_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2RGB)
    offset = window_size // 2
    y_range = input_image.shape[0] - offset
    x_range = input_image.shape[1] - offset

    dy, dx = np.gradient(input_image)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2

    for y in range(offset, y_range):
        for x in range(offset, x_range):

            # Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1

            # The variable names are representative to
            # the variable of the Harris corner equation
            windowIxx = Ixx[start_y: end_y, start_x: end_x]
            windowIxy = Ixy[start_y: end_y, start_x: end_x]
            windowIyy = Iyy[start_y: end_y, start_x: end_x]

            # Sum of squares of intensities of partial derevatives
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy

            # Calculate r for Harris Corner equation
            r = det - alpha * (trace ** 2)

            if r > threshold:
                corner_list.append([x, y, r])
                output_image[y, x] = (0, 0, 255)

    return corner_list, output_image
