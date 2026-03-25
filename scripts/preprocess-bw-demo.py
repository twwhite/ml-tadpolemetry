"""Quick preprocessing visualizer — original with binary B&W overlaid at adjustable opacity."""

import sys
import time

import cv2

WINDOW_NAME = "original + binary overlay"
TOGGLE_DELAY = 0.5

img = cv2.imread(sys.argv[1])

OTSU_OFFSET = 50  # increase to be less aggressive
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binary = cv2.threshold(gray, otsu_threshold + OTSU_OFFSET, 255, cv2.THRESH_BINARY)
binary_3channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 800)

opacities = [0.0, 1.0]
i = 0

while True:
    opacity = opacities[i % 2]
    overlay = cv2.addWeighted(img, 1.0, binary_3channel, opacity, 0)
    cv2.imshow(WINDOW_NAME, overlay)

    if cv2.waitKey(int(TOGGLE_DELAY * 1000)) != -1:
        break

    i += 1

cv2.destroyAllWindows()
