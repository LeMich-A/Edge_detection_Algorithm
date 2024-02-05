from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

# Defines a function ransac_line_fit that takes a set of points and fits a line using RANSAC
def ransac_line_fit(points):
    _, _, vx, vy = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)#Uses the cv.fitLine method to fit a line to the input points using the RANSAC algorithm
    return vx, vy

def draw_ransac_lines(drawing, lines):#Defines a function draw_ransac_lines that takes an image and a list of lines and draws the lines on the image.
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(drawing, (x1, y1), (x2, y2), (0, 0, 255), 2)

def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    lines = []
    for contour in contours:
        if len(contour) > 50:  # Choose a minimum number of points for RANSAC
            epsilon = 0.01 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            lines.append(cv.fitLine(approx, cv.DIST_L2, 0, 0.01, 0.01))

    draw_ransac_lines(drawing, lines)

    cv.imshow('RANSAC Lines', drawing)
    cv.imwrite('ransac_lines.png', drawing)

parser = argparse.ArgumentParser(description='Code for RANSAC Lines tutorial.')
parser.add_argument('--input', help='Path to input image.', default=r'/home/mich/Documents/Michel/Opencv/Tests/siccwhip.jpg')
args = parser.parse_args()
src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

max_thresh = 255
thresh = 150  # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()
