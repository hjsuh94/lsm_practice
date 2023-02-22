import numpy as np
import cv2

img = cv2.imread("bunny.png")
img = cv2.resize(img, (64, 64))
img = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=[255,255,255])
img = cv2.inRange(img, (235, 0, 0), (255, 255, 255))
img[60:68, 47:57] = 0

cv2.imwrite("bunny_proc.png", img)
